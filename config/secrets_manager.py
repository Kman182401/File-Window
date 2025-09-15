"""
Secure Secrets Management System for Trading Platform

This module provides a centralized, secure way to manage API keys, credentials,
and sensitive configuration data. It implements multiple layers of security:
- Environment variable loading with validation
- AWS Secrets Manager integration for production
- Encrypted local cache for development
- Automatic credential rotation support
- Audit logging for all credential access

Author: AI Trading System
Version: 2.0.0
Security Level: HIGH
"""

import os
import json
import logging
import hashlib
import base64
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
import warnings
import hmac
import secrets

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    warnings.warn("AWS SDK not available. AWS Secrets Manager integration disabled.")

logger = logging.getLogger(__name__)


class SecretValidationError(Exception):
    """Raised when a secret fails validation"""
    pass


class SecretNotFoundError(Exception):
    """Raised when a required secret is not found"""
    pass


class SecretRotationRequired(Exception):
    """Raised when a secret needs rotation"""
    pass


class SecretValidator:
    """Validates secrets based on their type and requirements"""
    
    @staticmethod
    def validate_api_key(key: str, key_name: str) -> bool:
        """
        Validate API key format and structure
        
        Args:
            key: The API key to validate
            key_name: Name of the key for specific validation rules
            
        Returns:
            bool: True if valid
            
        Raises:
            SecretValidationError: If validation fails
        """
        if not key or not isinstance(key, str):
            raise SecretValidationError(f"{key_name}: API key must be a non-empty string")
        
        # Remove any whitespace
        key = key.strip()
        
        # Check minimum length
        if len(key) < 10:
            raise SecretValidationError(f"{key_name}: API key too short")
        
        # Check for common placeholder values
        placeholders = ['your-api-key', 'api-key-here', 'xxx', 'test', 'demo']
        if any(placeholder in key.lower() for placeholder in placeholders):
            raise SecretValidationError(f"{key_name}: API key appears to be a placeholder")
        
        # Specific validation rules per service
        validation_rules = {
            'GOOGLE_CSE_API_KEY': lambda k: k.startswith('AIza'),
            'MARKETAUX_API_KEY': lambda k: len(k) == 36 and '-' in k,  # UUID format
            'ALPHAVANTAGE_API_KEY': lambda k: len(k) >= 16,
            'AWS_ACCESS_KEY_ID': lambda k: k.startswith('AKIA') and len(k) == 20,
            'AWS_SECRET_ACCESS_KEY': lambda k: len(k) == 40
        }
        
        if key_name in validation_rules:
            if not validation_rules[key_name](key):
                raise SecretValidationError(f"{key_name}: Invalid format for this service")
        
        return True
    
    @staticmethod
    def validate_connection_params(params: Dict[str, Any]) -> bool:
        """
        Validate connection parameters
        
        Args:
            params: Dictionary of connection parameters
            
        Returns:
            bool: True if valid
            
        Raises:
            SecretValidationError: If validation fails
        """
        required_fields = ['host', 'port']
        
        for field in required_fields:
            if field not in params:
                raise SecretValidationError(f"Missing required connection parameter: {field}")
        
        # Validate host
        host = params.get('host', '')
        if not host or host in ['0.0.0.0', '255.255.255.255']:
            raise SecretValidationError(f"Invalid host: {host}")
        
        # Validate port
        try:
            port = int(params.get('port', 0))
            if not 1 <= port <= 65535:
                raise SecretValidationError(f"Port must be between 1 and 65535: {port}")
        except (ValueError, TypeError):
            raise SecretValidationError(f"Invalid port: {params.get('port')}")
        
        return True


class LocalEncryption:
    """Handles local obfuscation of secrets for development environments
    
    Note: This provides basic obfuscation, not cryptographic security.
    For production, use AWS Secrets Manager or similar.
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize local encryption
        
        Args:
            master_key: Master encryption key. If None, generates from machine ID
        """
        self.master_key = master_key or self._generate_machine_key()
    
    def _generate_machine_key(self) -> bytes:
        """Generate a machine-specific encryption key"""
        # Combine multiple machine-specific attributes
        machine_id = f"{os.getuid()}-{os.getgid()}-{os.uname().nodename}"
        
        # Use SHA256 to derive a key
        key = hashlib.sha256(machine_id.encode() + b'trading-system-salt').digest()
        return key
    
    def _xor_cipher(self, data: bytes, key: bytes) -> bytes:
        """Simple XOR cipher for obfuscation"""
        # Extend key to match data length
        extended_key = key * (len(data) // len(key) + 1)
        return bytes(a ^ b for a, b in zip(data, extended_key))
    
    def encrypt(self, data: str) -> str:
        """Obfuscate a string"""
        data_bytes = data.encode('utf-8')
        # Add a simple checksum for integrity
        checksum = hashlib.md5(data_bytes).digest()
        payload = checksum + data_bytes
        
        # XOR cipher
        encrypted = self._xor_cipher(payload, self.master_key)
        
        # Base64 encode for storage
        return base64.b64encode(encrypted).decode('ascii')
    
    def decrypt(self, encrypted_data: str) -> str:
        """De-obfuscate a string"""
        try:
            # Base64 decode
            encrypted = base64.b64decode(encrypted_data.encode('ascii'))
            
            # XOR decipher
            payload = self._xor_cipher(encrypted, self.master_key)
            
            # Verify checksum
            checksum = payload[:16]
            data_bytes = payload[16:]
            
            if hashlib.md5(data_bytes).digest() != checksum:
                raise ValueError("Checksum verification failed")
            
            return data_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")


class SecretsManager:
    """
    Centralized secrets management with multiple backend support
    
    This class provides a unified interface for managing secrets across
    different environments (development, staging, production) with
    appropriate security measures for each.
    """
    
    def __init__(self, 
                 environment: str = 'development',
                 aws_region: str = 'us-east-1',
                 cache_ttl: int = 3600,
                 enable_rotation: bool = True,
                 audit_access: bool = True):
        """
        Initialize the secrets manager
        
        Args:
            environment: Current environment (development/staging/production)
            aws_region: AWS region for Secrets Manager
            cache_ttl: Cache time-to-live in seconds
            enable_rotation: Enable automatic secret rotation checks
            audit_access: Enable audit logging of secret access
        """
        self.environment = environment
        self.aws_region = aws_region
        self.cache_ttl = cache_ttl
        self.enable_rotation = enable_rotation
        self.audit_access = audit_access
        
        # Initialize components
        self.validator = SecretValidator()
        self.local_encryption = LocalEncryption()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._rotation_schedule: Dict[str, datetime] = {}
        
        # Initialize AWS client if available and in production
        self.aws_client = None
        if AWS_AVAILABLE and environment == 'production':
            try:
                self.aws_client = boto3.client('secretsmanager', region_name=aws_region)
                logger.info(f"AWS Secrets Manager initialized for region {aws_region}")
            except NoCredentialsError:
                logger.warning("AWS credentials not found. Falling back to environment variables.")
        
        # Load secrets configuration
        self.secrets_config = self._load_secrets_config()
        
        # Initialize audit log
        if self.audit_access:
            self._init_audit_log()
    
    def _load_secrets_config(self) -> Dict[str, Any]:
        """Load secrets configuration file"""
        config_path = Path(__file__).parent / 'secrets_config.json'
        
        # Default configuration
        default_config = {
            'api_keys': {
                'GOOGLE_CSE_API_KEY': {
                    'required': False,
                    'rotation_days': 90,
                    'fallback': 'disable_feature'
                },
                'MARKETAUX_API_KEY': {
                    'required': False,
                    'rotation_days': 365,
                    'fallback': 'disable_feature'
                },
                'ALPHAVANTAGE_API_KEY': {
                    'required': False,
                    'rotation_days': 365,
                    'fallback': 'disable_feature'
                }
            },
            'connections': {
                'IBKR': {
                    'required': True,
                    'parameters': ['host', 'port', 'client_id'],
                    'defaults': {
                        'host': '127.0.0.1',
                        'port': 4004,
                        'client_id': 9002
                    }
                }
            },
            'aws': {
                'S3_BUCKET': {
                    'required': True,
                    'default': 'omega-singularity-ml'
                }
            }
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load secrets config: {e}. Using defaults.")
        
        return default_config
    
    def _init_audit_log(self):
        """Initialize audit logging for secret access"""
        audit_dir = Path.home() / 'logs' / 'security'
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        audit_handler = logging.FileHandler(
            audit_dir / f'secrets_audit_{datetime.now():%Y%m%d}.log'
        )
        audit_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        self.audit_logger = logging.getLogger('secrets_audit')
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)
    
    def _log_access(self, secret_name: str, success: bool, source: str):
        """Log secret access for audit purposes"""
        if self.audit_access and hasattr(self, 'audit_logger'):
            self.audit_logger.info(
                f"Secret access: {secret_name} | Success: {success} | "
                f"Source: {source} | Environment: {self.environment}"
            )
    
    @lru_cache(maxsize=32)
    def get_api_key(self, key_name: str, validate: bool = True) -> Optional[str]:
        """
        Retrieve an API key from the most appropriate source
        
        Args:
            key_name: Name of the API key
            validate: Whether to validate the key format
            
        Returns:
            The API key or None if not found and not required
            
        Raises:
            SecretNotFoundError: If a required key is not found
            SecretValidationError: If validation fails
        """
        # Check cache first
        if key_name in self._cache:
            cache_entry = self._cache[key_name]
            if datetime.now() < cache_entry['expires']:
                self._log_access(key_name, True, 'cache')
                return cache_entry['value']
        
        api_key = None
        source = 'not_found'
        
        # Try AWS Secrets Manager first (production only)
        if self.aws_client and self.environment == 'production':
            try:
                api_key = self._get_from_aws(key_name)
                source = 'aws_secrets_manager'
            except Exception as e:
                logger.debug(f"AWS Secrets Manager lookup failed for {key_name}: {e}")
        
        # Try environment variables
        if not api_key:
            api_key = os.getenv(key_name)
            if api_key:
                source = 'environment'
        
        # Try local encrypted file (development only)
        if not api_key and self.environment == 'development':
            try:
                api_key = self._get_from_local_encrypted(key_name)
                source = 'local_encrypted'
            except Exception as e:
                logger.debug(f"Local encrypted lookup failed for {key_name}: {e}")
        
        # Check if this key is required
        key_config = self.secrets_config.get('api_keys', {}).get(key_name, {})
        if not api_key and key_config.get('required', False):
            self._log_access(key_name, False, source)
            raise SecretNotFoundError(
                f"Required API key '{key_name}' not found. "
                f"Please set it as an environment variable or in AWS Secrets Manager."
            )
        
        # Validate if requested and key was found
        if api_key and validate:
            try:
                self.validator.validate_api_key(api_key, key_name)
            except SecretValidationError as e:
                self._log_access(key_name, False, f"{source}_validation_failed")
                raise
        
        # Cache the result
        if api_key:
            self._cache[key_name] = {
                'value': api_key,
                'expires': datetime.now() + timedelta(seconds=self.cache_ttl)
            }
            
            # Check rotation if enabled
            if self.enable_rotation:
                self._check_rotation(key_name, key_config)
        
        self._log_access(key_name, api_key is not None, source)
        return api_key
    
    def get_connection_params(self, service: str) -> Dict[str, Any]:
        """
        Get connection parameters for a service
        
        Args:
            service: Service name (e.g., 'IBKR')
            
        Returns:
            Dictionary of connection parameters
        """
        config = self.secrets_config.get('connections', {}).get(service, {})
        params = {}
        
        # Get each parameter from environment or use default
        for param in config.get('parameters', []):
            env_key = f"{service}_{param.upper()}"
            value = os.getenv(env_key)
            
            if value is None:
                # Use default if available
                if 'defaults' in config and param in config['defaults']:
                    value = config['defaults'][param]
                elif config.get('required', False):
                    raise SecretNotFoundError(f"Required parameter {env_key} not found")
            
            if value is not None:
                # Convert port to int if applicable
                if param == 'port':
                    value = int(value)
                params[param] = value
        
        # Validate connection parameters
        if params:
            self.validator.validate_connection_params(params)
        
        self._log_access(f"{service}_connection", bool(params), 'environment')
        return params
    
    def _get_from_aws(self, secret_name: str) -> Optional[str]:
        """Retrieve secret from AWS Secrets Manager"""
        if not self.aws_client:
            return None
        
        try:
            response = self.aws_client.get_secret_value(SecretId=secret_name)
            
            if 'SecretString' in response:
                secret = response['SecretString']
                # Try to parse as JSON
                try:
                    secret_dict = json.loads(secret)
                    # If it's a dict, look for the key name or 'value'
                    return secret_dict.get(secret_name, secret_dict.get('value'))
                except json.JSONDecodeError:
                    # If not JSON, return as-is
                    return secret
            
            return None
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                return None
            raise
    
    def _get_from_local_encrypted(self, key_name: str) -> Optional[str]:
        """Retrieve secret from local encrypted file"""
        secrets_file = Path.home() / '.trading_secrets.enc'
        
        if not secrets_file.exists():
            return None
        
        try:
            with open(secrets_file, 'r') as f:
                encrypted_data = f.read()
            
            decrypted = self.local_encryption.decrypt(encrypted_data)
            secrets = json.loads(decrypted)
            
            return secrets.get(key_name)
        except Exception as e:
            logger.debug(f"Failed to read local encrypted secrets: {e}")
            return None
    
    def save_to_local_encrypted(self, secrets: Dict[str, str]):
        """
        Save secrets to local encrypted file (development only)
        
        Args:
            secrets: Dictionary of secrets to save
        """
        if self.environment != 'development':
            raise PermissionError("Local encrypted storage only available in development")
        
        secrets_file = Path.home() / '.trading_secrets.enc'
        
        # Load existing secrets
        existing = {}
        if secrets_file.exists():
            try:
                with open(secrets_file, 'r') as f:
                    encrypted_data = f.read()
                existing = json.loads(self.local_encryption.decrypt(encrypted_data))
            except Exception:
                pass
        
        # Update with new secrets
        existing.update(secrets)
        
        # Encrypt and save
        encrypted = self.local_encryption.encrypt(json.dumps(existing))
        
        with open(secrets_file, 'w') as f:
            f.write(encrypted)
        
        # Set restrictive permissions
        os.chmod(secrets_file, 0o600)
        
        logger.info(f"Saved {len(secrets)} secrets to local encrypted storage")
    
    def _check_rotation(self, key_name: str, config: Dict[str, Any]):
        """Check if a secret needs rotation"""
        if not config.get('rotation_days'):
            return
        
        # Check rotation schedule
        if key_name in self._rotation_schedule:
            next_rotation = self._rotation_schedule[key_name]
            if datetime.now() > next_rotation:
                logger.warning(
                    f"Secret '{key_name}' is due for rotation. "
                    f"Last rotation: {next_rotation - timedelta(days=config['rotation_days'])}"
                )
    
    def rotate_secret(self, key_name: str, new_value: str):
        """
        Rotate a secret with a new value
        
        Args:
            key_name: Name of the secret to rotate
            new_value: New secret value
        """
        # Validate new value
        self.validator.validate_api_key(new_value, key_name)
        
        # Update in appropriate backend
        if self.aws_client and self.environment == 'production':
            try:
                self.aws_client.put_secret_value(
                    SecretId=key_name,
                    SecretString=new_value
                )
                logger.info(f"Rotated secret '{key_name}' in AWS Secrets Manager")
            except ClientError as e:
                logger.error(f"Failed to rotate secret in AWS: {e}")
                raise
        else:
            # For non-production, update environment variable
            os.environ[key_name] = new_value
            logger.info(f"Updated secret '{key_name}' in environment")
        
        # Clear cache
        if key_name in self._cache:
            del self._cache[key_name]
        
        # Update rotation schedule
        config = self.secrets_config.get('api_keys', {}).get(key_name, {})
        if config.get('rotation_days'):
            self._rotation_schedule[key_name] = (
                datetime.now() + timedelta(days=config['rotation_days'])
            )
        
        self._log_access(key_name, True, 'rotation')
    
    def get_all_secrets(self) -> Dict[str, Any]:
        """
        Get all configured secrets (non-sensitive summary only)
        
        Returns:
            Dictionary with secret names and their status
        """
        summary = {
            'environment': self.environment,
            'api_keys': {},
            'connections': {},
            'aws': {}
        }
        
        # Check API keys
        for key_name, config in self.secrets_config.get('api_keys', {}).items():
            try:
                key = self.get_api_key(key_name, validate=False)
                summary['api_keys'][key_name] = {
                    'configured': key is not None,
                    'required': config.get('required', False),
                    'length': len(key) if key else 0
                }
            except Exception as e:
                summary['api_keys'][key_name] = {
                    'configured': False,
                    'error': str(e)
                }
        
        # Check connections
        for service in self.secrets_config.get('connections', {}).keys():
            try:
                params = self.get_connection_params(service)
                summary['connections'][service] = {
                    'configured': bool(params),
                    'parameters': list(params.keys()) if params else []
                }
            except Exception as e:
                summary['connections'][service] = {
                    'configured': False,
                    'error': str(e)
                }
        
        return summary


# Singleton instance for easy import
_secrets_manager_instance = None


def get_secrets_manager(**kwargs) -> SecretsManager:
    """
    Get or create the singleton SecretsManager instance
    
    Args:
        **kwargs: Arguments to pass to SecretsManager constructor
        
    Returns:
        SecretsManager instance
    """
    global _secrets_manager_instance
    
    if _secrets_manager_instance is None:
        # Detect environment
        environment = os.getenv('TRADING_ENVIRONMENT', 'development')
        
        # Create instance with environment-specific settings
        _secrets_manager_instance = SecretsManager(
            environment=environment,
            **kwargs
        )
        
        logger.info(f"SecretsManager initialized for {environment} environment")
    
    return _secrets_manager_instance


# Convenience functions for common operations
def get_api_key(key_name: str) -> Optional[str]:
    """Convenience function to get an API key"""
    return get_secrets_manager().get_api_key(key_name)


def get_ibkr_connection() -> Dict[str, Any]:
    """Convenience function to get IBKR connection parameters"""
    return get_secrets_manager().get_connection_params('IBKR')


def get_aws_bucket() -> str:
    """Convenience function to get the S3 bucket name"""
    return os.getenv('S3_BUCKET', 'omega-singularity-ml')


if __name__ == "__main__":
    # Self-test when run directly
    print("Trading System Secrets Manager - Self Test")
    print("=" * 50)
    
    manager = get_secrets_manager()
    summary = manager.get_all_secrets()
    
    print(f"\nEnvironment: {summary['environment']}")
    print("\nAPI Keys Status:")
    for key, status in summary['api_keys'].items():
        print(f"  {key}: {'✓' if status.get('configured') else '✗'} "
              f"(Required: {status.get('required', False)})")
    
    print("\nConnections Status:")
    for service, status in summary['connections'].items():
        print(f"  {service}: {'✓' if status.get('configured') else '✗'}")
    
    print("\nIBKR Connection Parameters:")
    try:
        ibkr_params = get_ibkr_connection()
        for key, value in ibkr_params.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"\nAWS S3 Bucket: {get_aws_bucket()}")
    print("\nSelf-test complete!")
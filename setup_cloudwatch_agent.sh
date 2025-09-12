#!/bin/bash

# Omega Singularity CloudWatch Agent Setup Script
# Production-grade, IMDSv2-compliant, modular, and auditable

LOG_FILE="/var/log/omegasingularity_cloudwatch_setup.log"
AGENT_CONFIG="/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

log "=== Omega Singularity CloudWatch Agent Setup Start ==="

# IAM Role Check (IMDSv2-compliant)
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s)
IAM_INFO=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s \
    http://169.254.169.254/latest/meta-data/iam/info)
echo "$IAM_INFO" | grep -q '"InstanceProfileArn"' || error_exit "Required IAM role not attached. Aborting."

log "IAM role detected and accessible."

# Ensure CloudWatch Agent is installed
if ! command -v /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl &> /dev/null; then
    log "CloudWatch Agent not found. Installing..."
    sudo apt-get update -y
    sudo apt-get install -y wget
    wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb -O /tmp/amazon-cloudwatch-agent.deb
    sudo dpkg -i /tmp/amazon-cloudwatch-agent.deb || error_exit "CloudWatch Agent install failed."
    log "CloudWatch Agent installed."
else
    log "CloudWatch Agent already installed."
fi

# Ensure agent config exists (customize path as needed)
if [ ! -f "$AGENT_CONFIG" ]; then
    error_exit "CloudWatch Agent config not found at $AGENT_CONFIG. Please create or provide a valid config."
fi

# Start/Restart CloudWatch Agent
log "Starting CloudWatch Agent..."
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a stop || true
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config -m ec2 -c file:"$AGENT_CONFIG" -s || error_exit "CloudWatch Agent start failed."

log "CloudWatch Agent started successfully."

# Extension hook: Add custom log/metric sources here

log "=== Omega Singularity CloudWatch Agent Setup Complete ==="

# Compliance/Audit: Human review marker
# [HUMAN_REVIEW_REQUIRED] Review log file at $LOG_FILE for errors or warnings.

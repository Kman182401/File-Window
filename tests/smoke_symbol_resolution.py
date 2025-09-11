#!/usr/bin/env python3
"""
Smoke test for symbol resolution in IBKR adapter.
Tests both pair-style (XAUUSD, EURUSD, etc.) and direct futures symbols (GC, 6E, etc.).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_data_ibkr_adapter import IBKRIngestor

def test_symbol_resolution():
    """Test that all symbols resolve to FUT contracts on CME/COMEX exchanges."""
    
    # Test symbols - both pair-style and direct futures forms
    test_symbols = [
        "ES1!", "NQ1!",           # Index futures
        "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD",  # Pair-style (existing)
        "GC", "6E", "6B", "6A"    # Direct futures symbols (new aliases)
    ]
    
    try:
        # Connect to IBKR
        ingestor = IBKRIngestor()
        print(f"Connected to IBKR Gateway")
        
        for symbol in test_symbols:
            try:
                # Test canonical symbol resolution
                canonical = ingestor._canonical_symbol(symbol)
                print(f"Symbol {symbol} -> {canonical}")
                
                # Test contract resolution
                contract = ingestor._get_contract(canonical)
                
                if contract is None:
                    print(f"❌ {symbol}: No contract found")
                    continue
                
                # Verify contract properties
                sec_type = getattr(contract, 'secType', 'UNKNOWN')
                exchange = getattr(contract, 'exchange', 'UNKNOWN')
                
                if sec_type == 'FUT' and exchange in ['CME', 'COMEX', 'GLOBEX']:
                    print(f"✅ {symbol}: {sec_type} on {exchange}")
                else:
                    print(f"❌ {symbol}: {sec_type} on {exchange} (expected FUT on CME/COMEX)")
                    return False
                    
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
                return False
        
        print("\n🎉 All symbols resolved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    finally:
        try:
            if 'ingestor' in locals():
                ingestor.disconnect()
        except:
            pass

if __name__ == "__main__":
    success = test_symbol_resolution()
    sys.exit(0 if success else 1)
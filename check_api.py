from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Binance Futures Testnet Connection Test")
print("=" * 60)

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print("[ERROR] API keys not found in .env")
    exit(1)

print(f"API Key: {api_key[:10]}...")
print(f"Secret: {api_secret[:10]}...")
print()

try:
    # Create testnet client
    print("Creating client with testnet=True...")
    client = Client(api_key, api_secret, testnet=True)

    print("Testing ping...")
    result = client.futures_ping()
    print(f"[OK] Ping: {result}")

    print("Getting account balance...")
    balance = client.futures_account_balance()
    print(f"[OK] Account info retrieved")

    for asset in balance:
        if float(asset['balance']) > 0:
            print(f"  {asset['asset']}: {asset['balance']}")

    print("\n[SUCCESS] All tests passed!")
    print("Your API keys are working correctly.")

except Exception as e:
    print(f"\n[ERROR] {e}")
    print("\nPlease check:")
    print("1. API keys are from https://testnet.binancefuture.com")
    print("2. 'Enable Futures' is checked in API settings")
    print("3. Keys are copied correctly (no extra spaces)")

print("=" * 60)
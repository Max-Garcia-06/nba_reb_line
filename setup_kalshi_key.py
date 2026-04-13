"""
setup_kalshi_key.py
-------------------
Run this once after creating a new API key on kalshi.com.
It stores both values atomically and verifies the connection works.

Usage:
    python3 setup_kalshi_key.py
"""

import base64
import os
import sys
import time
from pathlib import Path

def main():
    print("=== Kalshi API Key Setup ===\n")
    print("Go to kalshi.com → Settings → API → Create Key (Read/Write)")
    print("You will see TWO values. Have them ready.\n")

    # 1. Key ID
    key_id = input("Paste the Key ID (short UUID, e.g. abc123-...): ").strip()
    if not key_id:
        print("Error: Key ID cannot be empty.")
        sys.exit(1)

    # 2. Private key — read from file to avoid terminal paste corruption
    print("\nSave the RSA Private Key to a temporary file first, then enter the path.")
    print("Tip: In the Kalshi dashboard, click 'Download' or copy → open TextEdit → paste → save as private_key.pem\n")
    key_file = input("Path to the private key file (e.g. ~/Downloads/private_key.pem): ").strip()
    key_file = os.path.expanduser(key_file)

    if not os.path.exists(key_file):
        print(f"Error: File not found: {key_file}")
        sys.exit(1)

    with open(key_file, "r") as f:
        private_key_pem = f.read().strip()

    if "BEGIN" not in private_key_pem:
        print("Error: That doesn't look like a PEM key.")
        sys.exit(1)

    # 3. Verify the key loads
    try:
        from cryptography.hazmat.primitives import serialization
        priv = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
        print("\nPrivate key loaded successfully.")
    except Exception as e:
        print(f"\nError loading private key: {e}")
        sys.exit(1)

    # 4. Save private key
    key_dir = Path.home() / ".kalshi"
    key_dir.mkdir(exist_ok=True)
    key_path = key_dir / "private_key.pem"
    key_path.write_text(private_key_pem + "\n")
    print(f"Private key saved to {key_path}")

    # 5. Update .env
    env_path = Path(__file__).parent / ".env"
    env_text = env_path.read_text()

    import re
    env_text = re.sub(r"KALSHI_API_KEY_ID=.*", f"KALSHI_API_KEY_ID={key_id}", env_text)
    env_text = re.sub(r"KALSHI_PRIVATE_KEY_PATH=.*", f"KALSHI_PRIVATE_KEY_PATH={key_path}", env_text)
    env_path.write_text(env_text)
    print(f".env updated with new Key ID: {key_id}")

    # 6. Verify against live API
    print("\nTesting connection to Kalshi...")
    try:
        import requests
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        from dotenv import load_dotenv
        load_dotenv(override=True)
        base_url = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")

        path = "/portfolio/balance"
        ts = int(time.time() * 1000)
        msg = f"{ts}GET/trade-api/v2{path}".encode()
        sig = base64.b64encode(priv.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )).decode()

        r = requests.get(f"{base_url}{path}", headers={
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "KALSHI-ACCESS-SIGNATURE": sig,
        }, timeout=10)

        if r.status_code == 200:
            data = r.json()
            balance = data.get("balance", {})
            available = float(balance.get("available_balance", 0)) / 100
            print(f"\n✓ Connection successful! Account balance: ${available:.2f}")
            print("\nYou're all set. Run the pipeline with:")
            print("  python3 run_pipeline.py scan --bankroll 200 --threshold 0.20 --one-per-player --max-signals 10")
        else:
            print(f"\n✗ Auth failed ({r.status_code}): {r.text[:200]}")
            print("The Key ID and private key don't match. Recreate the key on kalshi.com and run this script again.")
    except Exception as e:
        print(f"\nConnection error: {e}")

if __name__ == "__main__":
    main()

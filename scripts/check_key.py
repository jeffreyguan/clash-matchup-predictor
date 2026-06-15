#!/usr/bin/env python3
"""Quick health check for the Clash Royale API key — no full fetch.

Hits a cheap endpoint and reports OK / the exact failure. On an IP-lock
error it prints the current public IP so you know what to whitelist.
Exit code 0 = key works, 1 = key does not.
"""
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent

token = None
env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if line.startswith("CR_API_TOKEN="):
            token = line.split("=", 1)[1].strip()
            break

if not token:
    print("✗ No CR_API_TOKEN found in .env")
    sys.exit(1)

try:
    r = requests.get(
        "https://api.clashroyale.com/v1/cards",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
except requests.RequestException as e:
    print(f"✗ Network error: {e}")
    sys.exit(1)

if r.status_code == 200:
    print("✓ Key works — API reachable and authorized.")
    sys.exit(0)

reason = ""
try:
    body = r.json()
    reason = body.get("reason", "")
    message = body.get("message", "")
except Exception:
    message = r.text[:200]

print(f"✗ HTTP {r.status_code}  {reason}: {message}")

if reason == "accessDenied.invalidIp":
    try:
        ip = requests.get("https://api.ipify.org", timeout=10).text.strip()
        print(f"  → Whitelist this IP on the key: {ip}")
        print("    https://developer.clashroyale.com/#/account")
    except requests.RequestException:
        pass

sys.exit(1)

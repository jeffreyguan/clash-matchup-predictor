"""
Fetch card metadata from the Clash Royale API and save elixir costs.

Output: card_features.csv
  Columns: card_id, elixir_cost (normalized 0-1)
  card_id matches the remapped sequential IDs in card_map.csv.
"""

import csv
from pathlib import Path

import requests

env_path = Path(__file__).parent / ".env"
API_TOKEN = None
with open(env_path) as f:
    for line in f:
        if line.startswith("CR_API_TOKEN="):
            API_TOKEN = line.strip().split("=", 1)[1]
            break

if not API_TOKEN:
    raise RuntimeError("CR_API_TOKEN not found in .env")

BASE_URL = "https://api.clashroyale.com/v1"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

ELIXIR_MAX = 10.0  # normalize elixir costs to [0, 1]


def main():
    resp = requests.get(f"{BASE_URL}/cards", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    cards = {card["id"]: card for card in resp.json()["items"]}

    card_map_path = Path(__file__).parent / "card_map.csv"
    with open(card_map_path) as f:
        card_map = {int(row["original_id"]): int(row["card_id"]) for row in csv.DictReader(f)}

    output_path = Path(__file__).parent / "card_features.csv"
    missing = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["card_id", "elixir_cost"])
        for orig_id, card_id in sorted(card_map.items(), key=lambda x: x[1]):
            card = cards.get(orig_id)
            if card is None:
                elixir = 0.0
                missing += 1
            else:
                elixir = card.get("elixirCost", 0) / ELIXIR_MAX
            writer.writerow([card_id, elixir])

    print(f"Saved {len(card_map)} cards to {output_path}")
    if missing:
        print(f"Warning: {missing} cards not found in API response (elixir set to 0)")


if __name__ == "__main__":
    main()

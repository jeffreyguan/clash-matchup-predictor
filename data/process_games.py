"""
Process games.json into a compact dataset: card IDs (no support cards) + winner.

Output: processed_games.csv
  Columns: team_0..team_7, opp_0..opp_7, winner
  Card IDs are remapped to sequential integers (0-based, sorted by original ID).
  winner: 0 = team, 1 = opponent. Draws (trophyChange == 0) are skipped.

Output: card_map.csv
  Columns: original_id, card_id  (lookup table for the remapped integers)
"""

import csv
import json
from pathlib import Path


INPUT = Path(__file__).parent / "games.json"
OUTPUT = Path(__file__).parent / "processed_games.csv"
CARD_MAP_OUTPUT = Path(__file__).parent / "card_map.csv"


def determine_winner(battle):
    team_change = (battle.get("team") or [{}])[0].get("trophyChange", 0)
    if team_change > 0:
        return 0
    if team_change < 0:
        return 1
    return None  # draw


def get_card_ids(player):
    return [card["id"] for card in player.get("cards", [])]


def main():
    with open(INPUT, encoding="utf-8") as f:
        battles = json.load(f)

    # First pass: collect all unique card IDs
    all_ids: set[int] = set()
    for battle in battles:
        team = (battle.get("team") or [{}])[0]
        opponent = (battle.get("opponent") or [{}])[0]
        all_ids.update(get_card_ids(team))
        all_ids.update(get_card_ids(opponent))

    card_map: dict[int, int] = {cid: idx for idx, cid in enumerate(sorted(all_ids))}

    # Second pass: build processed game records
    games = []
    skipped_draws = 0
    skipped_missing = 0

    for battle in battles:
        winner = determine_winner(battle)
        if winner is None:
            skipped_draws += 1
            continue

        team = (battle.get("team") or [{}])[0]
        opponent = (battle.get("opponent") or [{}])[0]

        team_ids = get_card_ids(team)
        opp_ids = get_card_ids(opponent)

        if not team_ids or not opp_ids:
            skipped_missing += 1
            continue

        games.append(
            [card_map[cid] for cid in team_ids]
            + [card_map[cid] for cid in opp_ids]
            + [winner]
        )

    team_cols = [f"team_{i}" for i in range(8)]
    opp_cols = [f"opp_{i}" for i in range(8)]

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(team_cols + opp_cols + ["winner"])
        writer.writerows(games)

    with open(CARD_MAP_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_id", "card_id"])
        for orig, remapped in card_map.items():
            writer.writerow([orig, remapped])

    print(f"Cards:          {len(card_map)}")
    print(f"Games written:  {len(games)}")
    print(f"Draws skipped:  {skipped_draws}")
    print(f"Missing skipped:{skipped_missing}")
    print(f"Output:         {OUTPUT}")
    print(f"Card map:       {CARD_MAP_OUTPUT}")


if __name__ == "__main__":
    main()

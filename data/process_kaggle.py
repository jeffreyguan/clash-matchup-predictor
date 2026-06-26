"""
Process the s1m0n38/clash-royale-games Kaggle dataset into the training format.

The dataset is one CSV per day (no header). Each row is:
  0:timestamp 1:mode
  2:team_tag 3:team_trophies 4:team_crowns  5..12:team card ids
  13:opp_tag 14:opp_trophies 15:opp_crowns  16..23:opp card ids

Card ids are Supercell ids (26000000...), the same scheme as card_map.csv,
so they remap cleanly through the existing map. Battles containing a card not
in card_map.csv are skipped (keeps card_id alignment with card_features.csv).

Output: processed_games.csv  (team_0..7, opp_0..7, winner)
  winner: 0 = team won, 1 = opponent won. Draws (equal crowns) are skipped.
"""

import csv
import glob
from pathlib import Path


DATA_DIR = Path(__file__).parent
INPUT_GLOB = str(DATA_DIR / "kaggle" / "**" / "*.csv")  # recurse into date folders
CARD_MAP = DATA_DIR / "card_map.csv"
OUTPUT = DATA_DIR / "processed_games_s53.csv"  # separate from the API-derived file

TEAM_CARDS = slice(5, 13)
OPP_CARDS = slice(16, 24)
TEAM_CROWNS = 4
OPP_CROWNS = 15

# Cap on total games written. The season has ~40M rows, so we sample evenly
# across all days (every Nth row) rather than taking the first ones.
MAX_GAMES = 5_000_000


def load_card_map(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {int(r["original_id"]): int(r["card_id"]) for r in reader}


def determine_winner(team_crowns, opp_crowns):
    if team_crowns > opp_crowns:
        return 0
    if opp_crowns > team_crowns:
        return 1
    return None  # draw


def count_rows(files):
    """Fast newline count across all files (files have no header)."""
    total = 0
    for fp in files:
        with open(fp, "rb") as f:
            while chunk := f.read(1 << 20):
                total += chunk.count(b"\n")
    return total


def main():
    card_map = load_card_map(CARD_MAP)

    files = sorted(glob.glob(INPUT_GLOB, recursive=True))
    if not files:
        raise SystemExit(f"No CSVs found under {INPUT_GLOB} — download the dataset first.")

    # Stride for even, season-wide sampling: keep every Nth row across all days.
    total_rows = count_rows(files)
    stride = max(1, total_rows // MAX_GAMES) if MAX_GAMES else 1
    print(f"Total rows: {total_rows:,}  ->  sampling every {stride} row(s)")

    games = []
    seen = 0          # raw rows scanned (drives the stride)
    skipped_draws = 0
    skipped_unknown = 0

    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 24:
                    continue
                seen += 1
                if seen % stride != 0:
                    continue

                winner = determine_winner(int(row[TEAM_CROWNS]), int(row[OPP_CROWNS]))
                if winner is None:
                    skipped_draws += 1
                    continue

                raw = [int(c) for c in row[TEAM_CARDS]] + [int(c) for c in row[OPP_CARDS]]
                if any(cid not in card_map for cid in raw):
                    skipped_unknown += 1
                    continue

                games.append([card_map[cid] for cid in raw] + [winner])
                if MAX_GAMES is not None and len(games) >= MAX_GAMES:
                    break
        if MAX_GAMES is not None and len(games) >= MAX_GAMES:
            break

    header = [f"team_{i}" for i in range(8)] + [f"opp_{i}" for i in range(8)] + ["winner"]
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(games)

    print(f"Files read:        {len(files)}")
    print(f"Games written:     {len(games)}")
    print(f"Draws skipped:     {skipped_draws}")
    print(f"Unknown-card skip: {skipped_unknown}")
    print(f"Output:            {OUTPUT}")


if __name__ == "__main__":
    main()

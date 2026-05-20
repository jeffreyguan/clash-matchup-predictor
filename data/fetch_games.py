"""
Fetch Path of Legends games from Ultimate Champion players across all regions.

Strategy:
  1. Probe for the current/most-recent PoL season ID.
  2. Pull global top-1000 rankings (all UC by definition).
  3. Pull every regional/country ranking; cross-reference player profiles
     to verify the UC leagueNumber for players not already in the global set.
  4. Fetch battle logs, keep only PoL battles, deduplicate.
  5. Write data/games.json.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

env_path = Path(__file__).parent / "data" / ".env"
API_TOKEN = None
with open(env_path) as f:
    for line in f:
        if line.startswith("CR_API_TOKEN="):
            API_TOKEN = line.strip().split("=", 1)[1]
            break

if not API_TOKEN:
    raise RuntimeError("CR_API_TOKEN not found in data/.env")

BASE_URL = "https://api.clashroyale.com/v1"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

RATE_LIMIT_DELAY = 0.12    # ~8 req/s, safely under silver-tier limit
MAX_RETRIES = 4
RANKINGS_LIMIT = 1000      # max players per rankings call

# ── HTTP helper ───────────────────────────────────────────────────────────────

def api_get(path, params=None):
    url = BASE_URL + path
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                log.warning(f"Rate limited — sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code in (404, 503):
                return None
            log.debug(f"HTTP {resp.status_code} for {path}")
            time.sleep(1)
        except requests.RequestException as e:
            log.warning(f"Request error ({path}): {e}")
            time.sleep(2 ** attempt)
    return None


# ── Season detection ──────────────────────────────────────────────────────────

def probe_season_id():
    """
    Find the most recent active PoL season ID.
    Tries YYYY-MM format for the current and past two months;
    falls back to numeric season IDs counting from Sep 2022 = Season 1.
    """
    now = datetime.now()

    for delta in range(3):
        month = now.month - delta
        year = now.year
        if month <= 0:
            month += 12
            year -= 1
        sid = f"{year}-{month:02d}"
        data = api_get(
            f"/locations/global/pathoflegend/{sid}/rankings/players",
            params={"limit": 1},
        )
        if data and data.get("items"):
            log.info(f"Active season: {sid}")
            return sid
        time.sleep(RATE_LIMIT_DELAY)

    # Numeric fallback (Sep 2022 = Season 1)
    start_year, start_month = 2022, 9
    elapsed = (now.year - start_year) * 12 + (now.month - start_month) + 1
    for n in range(elapsed, max(elapsed - 4, 0), -1):
        sid = str(n)
        data = api_get(
            f"/locations/global/pathoflegend/{sid}/rankings/players",
            params={"limit": 1},
        )
        if data and data.get("items"):
            log.info(f"Active season (numeric): {sid}")
            return sid
        time.sleep(RATE_LIMIT_DELAY)

    return None


# ── Rankings ──────────────────────────────────────────────────────────────────

def fetch_rankings(location_id, season_id):
    loc = quote(str(location_id), safe="")
    data = api_get(
        f"/locations/{loc}/pathoflegend/{season_id}/rankings/players",
        params={"limit": RANKINGS_LIMIT},
    )
    return data.get("items", []) if data else []


def get_locations():
    locations = []
    after = None
    while True:
        params = {"limit": 200}
        if after:
            params["after"] = after
        data = api_get("/locations", params=params)
        if not data:
            break
        locations.extend(data.get("items", []))
        after = data.get("paging", {}).get("cursors", {}).get("after")
        if not after:
            break
        time.sleep(RATE_LIMIT_DELAY)
    return locations


# ── Player profile / league detection ────────────────────────────────────────

def get_player_profile(tag):
    return api_get(f"/players/{quote(tag, safe='')}")


def pol_league_number(profile):
    """
    Return the highest PoL leagueNumber seen in the player's profile.
    Checks both last and current season results and returns the max.
    """
    last = profile.get("lastPathOfLegendSeasonResult") or {}
    current = profile.get("currentPathOfLegendSeasonResult") or {}
    return max(last.get("leagueNumber", 0), current.get("leagueNumber", 0))


def detect_uc_league_number(global_top_tag):
    """
    Determine the UC leagueNumber by inspecting the global rank-1 player's
    profile. UC is the top league, so their leagueNumber IS the threshold.
    """
    profile = get_player_profile(global_top_tag)
    if not profile:
        return None
    ln = pol_league_number(profile)
    log.info(f"UC leagueNumber = {ln}  (derived from rank-1 player {global_top_tag})")
    return ln


# ── Battle log ────────────────────────────────────────────────────────────────

def get_battle_log(tag):
    data = api_get(f"/players/{quote(tag, safe='')}/battlelog")
    return data if isinstance(data, list) else []


def is_pol_battle(battle):
    btype = battle.get("type", "")
    if "pathoflegend" in btype.lower():
        return True
    gm = battle.get("gameMode", {}).get("name", "")
    return "LadderMatch" in gm or "ladderMatch" in gm


def battle_key(battle):
    t = battle.get("battleTime", "")
    p1 = (battle.get("team") or [{}])[0].get("tag", "")
    p2 = (battle.get("opponent") or [{}])[0].get("tag", "")
    return (t, frozenset([p1, p2]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    output_path = Path("data") / "games.json"
    output_path.parent.mkdir(exist_ok=True)

    # Step 1 — find current season
    season_id = probe_season_id()
    if not season_id:
        log.error("Could not determine active PoL season — aborting.")
        return

    # Step 2 — global top-1000 (definitely all UC)
    log.info("Fetching global PoL rankings…")
    global_players = fetch_rankings("global", season_id)
    if not global_players:
        log.error("No global rankings returned — check API key / season.")
        return

    # Determine UC leagueNumber from rank-1 player's profile
    time.sleep(RATE_LIMIT_DELAY)
    uc_league_num = detect_uc_league_number(global_players[0]["tag"])
    time.sleep(RATE_LIMIT_DELAY)

    # All global top players are UC; collect them without extra profile calls
    accepted: dict[str, dict] = {p["tag"]: p for p in global_players}
    log.info(f"Global rankings: {len(accepted)} players accepted")

    # Step 3 — regional rankings
    log.info("Fetching locations…")
    locations = get_locations()
    log.info(f"Found {len(locations)} locations; querying regional rankings…")

    new_tags: list[str] = []
    for loc in tqdm(locations, desc="Regional rankings"):
        for p in fetch_rankings(loc["id"], season_id):
            if p["tag"] not in accepted:
                new_tags.append(p["tag"])
                accepted[p["tag"]] = p   # tentatively add; verify below
        time.sleep(RATE_LIMIT_DELAY)

    log.info(f"Regional rankings added {len(new_tags)} candidate players (need UC verification)")

    # Verify regional-only players by fetching their profiles
    rejected = 0
    for tag in tqdm(new_tags, desc="Verifying UC status"):
        profile = get_player_profile(tag)
        if not profile:
            del accepted[tag]
            rejected += 1
            time.sleep(RATE_LIMIT_DELAY)
            continue
        ln = pol_league_number(profile)
        if uc_league_num and ln < uc_league_num:
            del accepted[tag]
            rejected += 1
        time.sleep(RATE_LIMIT_DELAY)

    log.info(
        f"After UC verification: {len(accepted)} players accepted, {rejected} rejected. "
        f"(UC = leagueNumber {uc_league_num})"
    )

    # Step 4 — fetch battle logs
    seen: set = set()
    battles: list[dict] = []
    skipped = 0

    for tag in tqdm(list(accepted.keys()), desc="Fetching battle logs"):
        for b in get_battle_log(tag):
            if not is_pol_battle(b):
                skipped += 1
                continue
            key = battle_key(b)
            if key not in seen:
                seen.add(key)
                battles.append(b)
        time.sleep(RATE_LIMIT_DELAY)

    log.info(f"PoL battles collected: {len(battles)} unique  ({skipped} non-PoL skipped)")

    # Step 5 — save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(battles, f, indent=2, ensure_ascii=False)

    log.info(f"Saved → {output_path}")


if __name__ == "__main__":
    main()

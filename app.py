import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src" / "finetune"))
sys.path.append(str(Path(__file__).parent / "src"))  # so model.py can import pretrain.model

import csv
import torch
import pandas as pd
import requests
import streamlit as st

from model import MatchupPredictor

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "checkpoints/pretrained_model_best.pth"


@st.cache_resource
def load_model_and_cards():
    card_map_df = pd.read_csv(DATA_DIR / "card_map.csv")
    num_cards = len(card_map_df)

    # pretrain_path=None: build the architecture only, then load the finetuned weights.
    model = MatchupPredictor(num_cards=num_cards)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Try to fetch card names from API
    orig_to_card_id = dict(zip(card_map_df["original_id"], card_map_df["card_id"]))
    card_names = {}  # card_id -> name

    env_path = Path(__file__).parent / ".env"
    api_token = None
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("CR_API_TOKEN="):
                    api_token = line.strip().split("=", 1)[1]
                    break

    if api_token:
        try:
            resp = requests.get(
                "https://api.clashroyale.com/v1/cards",
                headers={"Authorization": f"Bearer {api_token}"},
                timeout=10,
            )
            if resp.ok:
                for card in resp.json().get("items", []):
                    orig_id = card["id"]
                    if orig_id in orig_to_card_id:
                        card_names[orig_to_card_id[orig_id]] = card["name"]
        except Exception:
            pass

    # Fall back to "Card {id}" for any missing
    for card_id in range(num_cards):
        if card_id not in card_names:
            card_names[card_id] = f"Card {card_id}"

    return model, card_names, num_cards


def predict(model, blue_ids, red_ids):
    # Blue occupies the team slots (0-7), red the opponent slots (8-15).
    # The model is trained on winner labels where 1 = opponent wins, so
    # sigmoid(logit) is P(red wins); blue's win probability is the complement.
    x = torch.tensor([sorted(blue_ids) + sorted(red_ids)], dtype=torch.long)
    with torch.no_grad():
        logit = model(x)
    red_win_prob = torch.sigmoid(logit).item()
    return 1 - red_win_prob


def main():
    st.set_page_config(page_title="CR Matchup Predictor", layout="wide")
    st.title("Clash Royale Matchup Predictor")

    model, card_names, num_cards = load_model_and_cards()

    options = [card_names[i] for i in range(1, num_cards)]  # skip card 0 (mask token, not a real card)
    name_to_id = {v: k for k, v in card_names.items()}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Blue Team")
        blue_selection = st.multiselect(
            "Select 8 cards",
            options=options,
            max_selections=8,
            key="blue",
        )

    with col2:
        st.subheader("Red Team")
        red_selection = st.multiselect(
            "Select 8 cards",
            options=options,
            max_selections=8,
            key="red",
        )

    blue_ready = len(blue_selection) == 8
    red_ready = len(red_selection) == 8

    if st.button("Predict", disabled=not (blue_ready and red_ready)):
        blue_ids = [name_to_id[n] for n in blue_selection]
        red_ids = [name_to_id[n] for n in red_selection]
        prob = predict(model, blue_ids, red_ids)

        st.divider()
        left, mid, right = st.columns([2, 1, 2])
        with left:
            st.metric("Blue Win Probability", f"{prob:.1%}")
        with mid:
            st.markdown("<h2 style='text-align:center;margin-top:8px'>vs</h2>", unsafe_allow_html=True)
        with right:
            st.metric("Red Win Probability", f"{1 - prob:.1%}")

        st.progress(prob)
    elif not (blue_ready and red_ready):
        missing = []
        if not blue_ready:
            missing.append(f"Blue needs {8 - len(blue_selection)} more card(s)")
        if not red_ready:
            missing.append(f"Red needs {8 - len(red_selection)} more card(s)")
        st.caption(" · ".join(missing))


if __name__ == "__main__":
    main()

# Clash Royale Matchup Predictor

A model that predicts the winner of a Clash Royale match given two decks. I had made a previous version that was much simpler, but I wanted to use a transformer to see how that would affect performance. The goal was that using a transformer would allow the model to better understand the relationships between cards via attention. 

I pretrained the transformer an older dataset (https://www.kaggle.com/datasets/s1m0n38/clash-royale-games) by predicting a masked card using the 80/10/10 split used in BERT. I didn't want to use this dataset for the actual training run due to its age. With how often the Clash Royale meta shifts, I thought that certain matchup spreads might've changed. Another possible extension of this project would be to use this dataset for training as well and save the current season as a test dataset. I only used the last season (53), and only pulled 5 million matches from it. After pretraining, I got these numbers:
Test Loss:   1.9520
Test Accuracy: 0.5319

For my actual training data, I scraped the Clash Royale Developer API for a couple weeks (06/15/2026-07/01/2026) nightly and only kept games that occured in ultimate champion. This dataset had 170060 total matches when I trained. I only wanted to use games in the highest ranks because I thought that lower skill levels would have too much noise, hence making the data not usable. If you want an extension of this project, you could train a model on matches generally and see if there is an accuracy difference (I would guess it would just be around 50%). After finetuning (stopped at epoch 25 with patience 5), I got these numbers:
```bash
Test Loss:   0.6540
Test Accuracy: 0.6082
```

To see if pretraining made a difference, I also trained a model that didn't use the pretrained weights. I got these numbers (stopped at epoch 24):
```bash
Test Loss:   0.6595
Test Accuracy: 0.5973
```

Both accuracy and loss are slightly better with the pretrained one, so one can see a small improvement in performance with pretraining. Although the final difference in accuracy was around 1%, at the first epoch the difference was around 3%. The only reason I bothered to pretrain was because my dataset was too small; I suspect that training on a larger dataset wouldn't have required pretraining given the small parameter count of this model. 

Below I asked Claude to generate a guide on how to recreate this project.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```
CR_API_TOKEN=<clash royale api token>   # scraping battles + card names in the app
KAGGLE_API_TOKEN=<kaggle api token>     # downloading the pretraining dataset
```

## Recreate from scratch

Each training script uses paths relative to its own folder, so run it from there.

```bash
# 1. Data
python data/fetch_games.py            # scrape battles -> data/games.json
python data/process_games.py          # -> data/card_map.csv + data/processed_games.csv
# download the Kaggle battle dataset into data/kaggle/, then:
python data/process_kaggle.py         # -> data/processed_games_s53.csv (reuses card_map.csv)

# 2. Pretrain the deck encoder
cd src/pretrain && python train.py    # reads processed_games_s53.csv -> checkpoints/ckpt_best.pth

# 3. Finetune for matchup prediction
cd ../finetune && python train.py     # loads ckpt_best.pth -> model_best.pth
cp model_best.pth ../../checkpoints/pretrained_model_best.pth
```

`process_games.py` writes `processed_games.csv`; rename it per season (e.g. `processed_games_s84.csv`) to match the path in `src/finetune/train.py`.

## Run the app

```bash
streamlit run app.py
```

Needs `data/card_map.csv` and a finetuned model at `checkpoints/pretrained_model_best.pth`.

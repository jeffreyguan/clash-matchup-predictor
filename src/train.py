import sys
sys.path.append("..")
import torch
import csv
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import GameDataset
from model import MatchupModel

def augment(data):
    augmented = []
    for row in data:
        augmented.append(row)
        flipped = row[8:16] + row[0:8] + [1 - row[-1]]
        augmented.append(flipped)
    return augmented

def load_data(path, test_size=0.2, batch_size=128):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        data = [[int(v) for v in row] for row in reader]

    print(f"Total samples: {len(data)}")
    train_data, temp = train_test_split(data, test_size=test_size, random_state=67)
    val_data, test_data = train_test_split(temp, test_size=0.5, random_state=67)

    train_data = augment(train_data)

    train_dataset = GameDataset(train_data)
    val_dataset = GameDataset(val_data)
    test_dataset = GameDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    labels = [row[-1] for row in train_data]
    n_losses = sum(1 for l in labels if l == 0)
    n_wins   = sum(1 for l in labels if l == 1)
    pos_weight = torch.tensor([n_losses / n_wins])

    return train_loader, val_loader, test_loader, pos_weight

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    print(f"  Train Loss: {train_loss / len(dataloader):.4f}")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            test_loss += loss.item()
    avg_loss = test_loss / len(dataloader)
    print(f"  Val Loss:   {avg_loss:.4f}")
    correct = (preds.round() == y_batch).float().sum()
    accuracy = correct / len(y_batch)
    print(f"  Val Accuracy: {accuracy:.4f}")
    return avg_loss

def load_card_features(path):
    df = pd.read_csv(path).sort_values("card_id")
    return torch.tensor(df[["elixir_cost"]].values, dtype=torch.float)

if __name__ == "__main__":
    train_loader, val_loader, test_loader, pos_weight = load_data("../data/processed_games.csv", test_size=0.2, batch_size=32)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    card_features = load_card_features("../data/card_features.csv").to(device)
    model = MatchupModel(num_cards=len(pd.read_csv("../data/card_map.csv")), embedding_dim=8, card_features=card_features).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience, wait = 5, 0

    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        val_loss = test_loop(val_loader, model, loss_fn)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model_best.pth')
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {t+1}")
                break

    print("\nTesting on the test set...")
    model.load_state_dict(torch.load('model_best.pth'))
    test_loop(test_loader, model, loss_fn)

    torch.save(model.state_dict(), 'model.pth')
    print("Model saved!")

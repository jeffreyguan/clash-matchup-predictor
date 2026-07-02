import sys
sys.path.append("..")
import torch
import csv
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import FinetuneDataset
from model import MatchupPredictor

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

    train_dataset = FinetuneDataset(train_data)
    val_dataset = FinetuneDataset(val_data)
    test_dataset = FinetuneDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

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
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            test_loss += loss_fn(preds, y_batch).item()
            correct += ((preds > 0).float() == y_batch).sum().item()
            total += y_batch.numel()
    avg_loss = test_loss / len(dataloader)
    print(f"  Val Loss:   {avg_loss:.4f}")
    print(f"  Val Accuracy: {correct / total:.4f}")
    return avg_loss

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data("../../data/processed_games_s84.csv", test_size=0.2, batch_size=512)
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
    # model = MatchupPredictor(pretrain_path="../../checkpoints/ckpt_best.pth").to(device)
    model = MatchupPredictor().to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([
        {"params": model.embedding.parameters(),   "lr": 1e-4},
        {"params": model.transformer.parameters(), "lr": 1e-4},
        {"params": model.network.parameters(),     "lr": 1e-3},
    ], weight_decay=1e-4)

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

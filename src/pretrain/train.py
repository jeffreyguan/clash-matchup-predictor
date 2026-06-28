import torch
import csv
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import PretrainDataset
from model import DeckTransformer

def load_data(path, test_size=0.2, batch_size=128):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        data = [[int(v) for v in row] for row in reader]

    print(f"Total samples: {len(data)}")
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=67)

    train_dataset = PretrainDataset(train_data)
    test_dataset = PretrainDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    labels = [row[-1] for row in train_data]

    return train_loader, test_loader

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        preds = model(x_batch)
        loss = loss_fn(preds.reshape(-1, 122), y_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    print(f"  Train Loss: {train_loss / len(dataloader):.4f}")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = loss_fn(preds.reshape(-1, 122), y_batch.reshape(-1))
            test_loss += loss.item()

            masks = y_batch != -100
            correct += (preds.argmax(-1)[masks] == y_batch[masks]).sum().item()
            total += masks.sum().item()

    avg_loss = test_loss / len(dataloader)
    print(f"  Val Loss:   {avg_loss:.4f}")
    accuracy = correct / total
    print(f"  Val Accuracy: {accuracy:.4f}")
    return avg_loss

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
    train_loader, test_loader = load_data("../../data/processed_games_s53.csv")
    model = DeckTransformer().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
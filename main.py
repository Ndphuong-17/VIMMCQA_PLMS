
from dataset import CustomVIMMCQADataset
from transformers import AutoModelForMultipleChoice, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from model import VimMCQAModel
from tqdm import tqdm
import torch
from torch import nn
import json

import argparse

parser = argparse.ArgumentParser(description="My Custom Model")
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--train_csv', type=str, required=True)
parser.add_argument('--val_csv', type=str, required=True)
parser.add_argument('--test_csv', type=str, required=True)
parser.add_argument('--batch_size', type=int, default = 16)
parser.add_argument('--epochs', type=int, default = 20)
parser.add_argument('--train_val_metric', type=str, default = 'train_val_metrics.json')
parser.add_argument('--checkpoint', type=str, default = 'checkpoints.pth')
parser.add_argument('--result_output', type=str, default = 'results.json')

args = parser.parse_args()

# Set up Dataset, Dataloader
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

train_dataset = CustomVIMMCQADataset(args.train_csv, tokenizer)
val_dataset = CustomVIMMCQADataset(args.val_csv, tokenizer)
test_dataset = CustomVIMMCQADataset(args.test_csv, tokenizer)

print("Loading DataLoader")
train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True,  collate_fn = train_dataset.collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = val_dataset.collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle = False, collate_fn=test_dataset.collate_fn)

print('================')
print("Loading Dataloader succesfully")
print("Train Length", len(train_dataloader))
print("Val Length", len(val_dataloader))
print("Test Length", len(test_dataloader))

# Set up model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VimMCQAModel(model_name_or_path = args.model_name_or_path).to(device)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('================')
print("Model Params: ", params)

# Training Components Configuration
print('================')
print("Loading Training Components")
optimizer = torch.optim.Adam(params = model.parameters())
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# # Training and Eval
print('================')
print('Training and Eval')
torch.manual_seed(42)

train_losses, train_accies = [], []
val_losses, val_accies = [], []

train_losses, train_accies = [], []
val_losses, val_accies = [], []

for epoch in range(args.epochs):

    train_loss, val_loss = 0, 0
    train_acc, val_acc = 0, 0
    total_train_samples = 0
    total_val_samples = 0

    model.train()  # Training
    for sample in tqdm(train_dataloader):
        prediction = model(sample)
        loss = criterion(prediction, sample['label'].float().to(device))

        train_loss += loss.item()
        train_acc += (prediction.round() == sample['label'].float().to(device)).sum().item()
        total_train_samples += len(sample['label'])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Average training loss and accuracy
    train_loss /= total_train_samples  # Average loss over total samples
    train_acc /= total_train_samples    # Average accuracy over total samples

    model.eval()  # Evaluation
    with torch.no_grad():
        for sample in tqdm(val_dataloader):
            prediction = model(sample)
            loss = criterion(prediction, sample['label'].float().to(device))

            val_loss += loss.item()
            val_acc += (prediction.round() == sample['label'].float().to(device)).sum().item()
            total_val_samples += len(sample['label'])

    # Average validation loss and accuracy
    val_loss /= total_val_samples  # Average loss over total samples
    val_acc /= total_val_samples    # Average accuracy over total samples

    train_losses.append(train_loss)
    train_accies.append(train_acc)
    val_losses.append(val_loss)
    val_accies.append(val_acc)

    # Step the scheduler based on validation loss
    scheduler.step(val_loss)

    print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

item = {
    'Train' : {
        'loss' : train_losses,
        'acc' : train_accies
    },
    'Val' : {
        'loss' : val_losses,
        'acc' : val_accies
    }
}

# Saving training and evaluation phase
torch.save(model.state_dict(), args.checkpoint)
with open(args.train_val_metric, 'w') as f:
    json.dump(item, f)

print('================')
print('Inference')

model.eval()
y_true = torch.tensor([]).to(device)
y_pred = torch.tensor([]).to(device)

for sample in tqdm(test_dataloader):
  prediction = model(sample)
  y_pred = torch.cat((y_pred, prediction.round()))
  y_true = torch.cat((y_true, sample['label'].to(device)))

results = {
    'true' : y_true,
    'prediction' : y_pred,
}

with open(args.result_output, 'w') as f:
    json.dump(results, f)

print('The script did run successfully')

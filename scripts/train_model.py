import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from scripts.data_preprocessing import get_dataloaders # Assuming data_preprocessing.py is in 'scripts/'
from models.cnn_predictor import PhosphoPredictorCNN

# Configurations
SEQUENCE_LENGTH = 21
KERNEL_SIZE = 5      
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
NUM_EPOCHS = 30

# Device Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS (GPU) for computations.")
else:
    device = torch.device("cpu")
    print("Using CPU for computations.")

# Loading the dataloaders
train_loader, val_loader, test_loader, num_amino_acids, aa_to_int_map = \
    get_dataloaders(batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, force_reprocess=False)

# Instantiate the model
model = PhosphoPredictorCNN(num_amino_acids=num_amino_acids, sequence_length=SEQUENCE_LENGTH, kernel_size=KERNEL_SIZE).to(device)

# Defining the loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Model initialized!")


# Defining the training and validation loop
best_val_loss = float('inf')

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()

    running_loss = 0.0
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(sequences)

        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()

        optimizer.step()
        running_loss += loss.item() * sequences.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train loss: {epoch_train_loss:.4f}")

   
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item() * sequences.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation loss: {epoch_val_loss:.4f}")

        # Only save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_save_path = os.path.join(project_root, 'models', 'best_phospho_predictor.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved. Saving model to {model_save_path}.")

print("Training Complete!")

print("Starting test set evaluation...")

model_save_path = os.path.join(project_root, 'models', 'best_phospho_predictor.pth')
test_model = PhosphoPredictorCNN(num_amino_acids=num_amino_acids, sequence_length=SEQUENCE_LENGTH, kernel_size=KERNEL_SIZE).to(device)

try:
    test_model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"Loaded best model from {model_save_path}")
except FileNotFoundError:
    print(f"Error: Best model not found at {model_save_path}. Ensure model is saved after training.")
    exit()

test_model.eval()

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs = test_model(sequences)
        probabilities  = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.5).float()

        all_preds.extend(predictions.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds).flatten()
all_probs = np.array(all_probs).flatten()
all_labels = np.array(all_labels).flatten()

# Calculate and print metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)

print("--- Test Set Performance ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print("--------------------------")




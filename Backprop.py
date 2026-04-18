import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class ANFISModel(nn.Module):
    def __init__(self):
        super(ANFISModel, self).__init__()
        # 1. Trainable Membership Function Parameters (Centers of 5 Gaussian curves)
        # We initialize them at 0, 0.25, 0.5, 0.75, 1.0 but let them 'wiggle'
        self.tox_centers = nn.Parameter(torch.linspace(0, 1, 5))
        self.sent_centers = nn.Parameter(torch.linspace(0, 1, 5))
        self.sigma = nn.Parameter(torch.ones(5) * 0.15)

        # 2. FAM Table Optimization (The 25 rules)
        # Instead of manual 'High/Low', these are weights the model learns
        self.fam_weights = nn.Parameter(torch.randn(5, 5))

    def forward(self, tox_input, sent_input):
        # Fuzzification: Calculate membership grades
        mu_tox = torch.exp(-0.5 * torch.pow((tox_input.unsqueeze(1) - self.tox_centers) / self.sigma, 2))
        mu_sent = torch.exp(-0.5 * torch.pow((sent_input.unsqueeze(1) - self.sent_centers) / self.sigma, 2))

        # Rule Layer: Combine Toxicity and Sentiment (Product T-Norm)
        # This creates the 5x5 FAM grid for every input
        batch_size = tox_input.shape[0]
        rules = torch.bmm(mu_tox.unsqueeze(2), mu_sent.unsqueeze(1)) # (Batch, 5, 5)

        # Defuzzification: Weighted average using the FAM weights
        num = torch.sum(rules * self.fam_weights, dim=(1, 2))
        den = torch.sum(rules, dim=(1, 2)) + 1e-9
        return num / den

# --- TRAINING LOOP ---
def train_anfis('C:\Users\shaun\Downloads\youtoxic_english_1000.csv'):
    # Load your file (Assuming you added BERT scores 'tox_score' and 'sent_score' already)
    df = pd.read_csv('C:\Users\shaun\Downloads\youtoxic_english_1000.csv')
    
    # Target: 1 for Toxic, 0 for Not Toxic (from your IsToxic column)
    X = torch.tensor(df[['tox_score', 'sent_score']].values, dtype=torch.float32)
    y = torch.tensor(df['IsToxic'].map({True: 1.0, False: 0.0}).values, dtype=torch.float32)

    model = ANFISModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(X[:, 0], X[:, 1])
        loss = criterion(predictions, y)
        loss.backward() # BACKPROPAGATION: The 'Wiggling' happens here
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    return model
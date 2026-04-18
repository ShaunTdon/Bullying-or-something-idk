import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# 1. Define the ANFIS-style Model Structure
class ANFISModel(nn.Module):
    def __init__(self):
        super(ANFISModel, self).__init__()
        # Trainable Centers for 5 Membership Functions (Toxicity & Sentiment)
        self.tox_centers = nn.Parameter(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
        self.sent_centers = nn.Parameter(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
        self.sigma = nn.Parameter(torch.ones(5) * 0.15) # Trainable width of curves

        # The Optimized FAM Table (5x5 matrix of weights)
        self.fam_weights = nn.Parameter(torch.randn(5, 5))

    def forward(self, tox_input, sent_input):
        # Fuzzification (Gaussian Membership)
        mu_tox = torch.exp(-0.5 * torch.pow((tox_input.unsqueeze(1) - self.tox_centers) / self.sigma, 2))
        mu_sent = torch.exp(-0.5 * torch.pow((sent_input.unsqueeze(1) - self.sent_centers) / self.sigma, 2))

        # Rule Layer (Product of memberships to create the 5x5 FAM grid)
        # Resulting shape: (batch_size, 5, 5)
        rules = torch.bmm(mu_tox.unsqueeze(2), mu_sent.unsqueeze(1))

        # Defuzzification (Weighted Average)
        num = torch.sum(rules * self.fam_weights, dim=(1, 2))
        den = torch.sum(rules, dim=(1, 2)) + 1e-9
        return num / den

# 2. The Training Loop
def start_training():
    print("🚀 Loading prepared data...")
    df = pd.read_csv('prepared_data.csv')
    
    # Prepare inputs (X) and targets (y)
    X = torch.tensor(df[['tox_score', 'sent_score']].values, dtype=torch.float32)
    # Convert True/False labels to 1.0/0.0
    y = torch.tensor(df['IsToxic'].astype(float).values, dtype=torch.float32)

    model = ANFISModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("🧠 Training the Fuzzy Inference Engine (Backpropagation)...")
    for epoch in range(200):
        optimizer.zero_grad()
        predictions = model(X[:, 0], X[:, 1])
        loss = criterion(predictions, y)
        loss.backward() # This is the "wiggling" of the curves
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Training Loss: {loss.item():.4f}")

    # 3. Save the result
    torch.save(model.state_dict(), "anfis_weights.pth")
    print("\n✅ Success! Optimized weights saved as 'anfis_weights.pth'")

if __name__ == "__main__":
    start_training()
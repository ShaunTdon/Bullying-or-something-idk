import torch
import torch.nn as nn
import numpy as np

# This MUST match the structure used during training
class ANFISModel(nn.Module):
    def __init__(self):
        super(ANFISModel, self).__init__()
        self.tox_centers = nn.Parameter(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
        self.sent_centers = nn.Parameter(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
        self.sigma = nn.Parameter(torch.ones(5) * 0.15)
        self.fam_weights = nn.Parameter(torch.randn(5, 5))

    def forward(self, tox_input, sent_input):
        mu_tox = torch.exp(-0.5 * torch.pow((tox_input.unsqueeze(1) - self.tox_centers) / self.sigma, 2))
        mu_sent = torch.exp(-0.5 * torch.pow((sent_input.unsqueeze(1) - self.sent_centers) / self.sigma, 2))
        rules = torch.bmm(mu_tox.unsqueeze(2), mu_sent.unsqueeze(1))
        num = torch.sum(rules * self.fam_weights, dim=(1, 2))
        den = torch.sum(rules, dim=(1, 2)) + 1e-9
        return num / den

# Load the trained brain
model = ANFISModel()
try:
    model.load_state_dict(torch.load("anfis_weights.pth"))
    model.eval()
    print("✅ ANFIS Engine Loaded Successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load weights: {e}")

def get_fuzzy_risk(toxicity_score, sentiment_score):
    # Convert inputs to Tensors
    t_in = torch.tensor([float(toxicity_score)], dtype=torch.float32)
    s_in = torch.tensor([float(sentiment_score)], dtype=torch.float32)
    
    # Run through the Optimized ANFIS brain
    with torch.no_grad():
        risk_output = model(t_in, s_in).item()
    
    # Ensure the risk stays between 0 and 1
    risk_output = max(0, min(1, risk_output))
    
    return risk_output, toxicity_score, sentiment_score
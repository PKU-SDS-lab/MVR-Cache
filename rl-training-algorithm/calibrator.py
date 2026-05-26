import torch
import torch.nn as nn

class SigmoidCalibrator(nn.Module):
    def __init__(self, init_t=0.5, init_gamma=10.0):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(init_t))
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(init_gamma)))

    def logits(self, s):
        gamma = torch.exp(self.log_gamma)
        return gamma * (s - self.t)

    def forward(self, s):
        logits = self.logits(s)
        return logits, torch.sigmoid(logits)

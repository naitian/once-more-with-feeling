"""
Base model for speech emotion recognition
"""

import torch


class BaseSERModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def loss(self, x, y):
        raise NotImplementedError

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

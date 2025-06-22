import torch.nn as nn
import torch

class DigitGenerator(nn.Module):
    def __init__(self, noise_dim=64, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_emb(labels)
        x = torch.cat([noise, labels], dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)
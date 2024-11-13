import torch
import torch.nn.functional as F
from torch.optim import Adam
from scripts.utils import get_param

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = get_param(config, "learning_rate", 0.001)
        self.epochs = get_param(config, "epochs", 5)
        self.optimizer = Adam(model.parameters(), lr=self.learning_rate)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)}] Loss: {loss.item():.6f}")

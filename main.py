import torch
from scripts.utils import load_config, get_param
from scripts.loaders import get_data_loaders
from scripts.models import SimpleCNN
from scripts.trainers import Trainer
from scripts.tests import test

def main():
    # Load configuration
    config = load_config()

    # Set device
    device = torch.device(get_param(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Load data
    train_loader, test_loader = get_data_loaders(config)

    # Initialize model, trainer, and start training
    model = SimpleCNN(config).to(device)
    trainer = Trainer(model, train_loader, test_loader, device, config)
    trainer.train()

    # Test the model
    test(model, test_loader, device, config)

if __name__ == "__main__":
    main()
import torch
from scripts.utils import load_config
from scripts.loaders import get_pet_data_loaders, preprocess
from scripts.models import get_object_detection_model, YOLOPretrain, YOLOTrain, transfer_weights
from scripts.trainers import pretrain_on_imagenet
import scripts.trainers
from scripts.tests import test_model
from scripts.utils import get_param

def main():
    # Load configuration
    config = load_config()

    # Set device
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print("Training on device", device)

    # Proprocess data
    # input_root = "data/pets"
    # output_root = "data/pets/preprocessed"
    # preprocess(input_root, output_root)

    # Load Data
    train_loader, val_loader, test_loader = get_pet_data_loaders(config)

    # Initialize Model
    # model = get_object_detection_model(config).to(device)
    num_classes = get_param(config, "num_classes", 37)
    print(num_classes)
    pretrain_model = YOLOPretrain().to(device)
    train_model = YOLOTrain(config).to(device)

    # Pretrain model
    # if config.get("pretrain", False):
    print("Pretraining on ImageNet...")
    pretrain_on_imagenet(pretrain_model, config, device)

    # After pretraining, transfer weights
    transfer_weights(pretrain_model, train_model)

    # Train and Validate
    scripts.trainers.train_model(train_model, train_loader, val_loader, device, config)

    # Test
    test_model(train_model, test_loader, device, config)

if __name__ == "__main__":
    main()
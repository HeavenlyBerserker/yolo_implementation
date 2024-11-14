from tqdm import tqdm
import torch
from scripts.utils import get_param
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def compute_yolo_loss(outputs, targets, S, B, num_classes):
    """
    Compute YOLO loss based on model outputs and targets.

    Parameters:
    - outputs: Tensor of shape (batch_size, S, S, B * 5 + num_classes).
    - targets: List of dictionaries with keys 'boxes' and 'labels' for each sample in the batch.
    - S: Grid size (number of cells in one dimension).
    - B: Number of bounding boxes per grid cell.
    - num_classes: Number of object classes.

    Returns:
    - total_loss: Computed loss as a scalar tensor.
    """
    batch_size = outputs.size(0)
    lambda_coord = 5  # Weight for coordinate loss
    lambda_noobj = 0.5  # Weight for no-object confidence loss

    # Reshape outputs to separate bounding boxes and class predictions
    outputs = outputs.view(batch_size, S, S, B * 5 + num_classes)
    pred_boxes = outputs[..., :B * 5].view(batch_size, S, S, B, 5)  # (x, y, w, h, confidence) for each box
    pred_classes = outputs[..., B * 5:]  # Class probabilities for each grid cell

    total_loss = 0.0

    for i in range(batch_size):
        coord_loss = 0.0
        conf_loss = 0.0
        class_loss = 0.0
        target_boxes = targets[i]["boxes"]
        target_labels = targets[i]["labels"]

        # Each target box is assigned to the grid cell it falls into
        for box_idx in range(target_boxes.size(0)):
            # Compute the center of the ground truth box
            x_center = (target_boxes[box_idx, 0] + target_boxes[box_idx, 2]) / 2
            y_center = (target_boxes[box_idx, 1] + target_boxes[box_idx, 3]) / 2

            # Normalize to grid cell indices
            cell_x = min(int(x_center * S / outputs.size(2)), S - 1)  # Ensure cell_x is within [0, S-1]
            cell_y = min(int(y_center * S / outputs.size(3)), S - 1)  # Ensure cell_y is within [0, S-1]

            # Compute the width and height of the target box
            box_w = target_boxes[box_idx, 2] - target_boxes[box_idx, 0]
            box_h = target_boxes[box_idx, 3] - target_boxes[box_idx, 1]

            best_iou = 0
            best_box = 0

            # Find the best bounding box for this cell (highest IOU)
            for b in range(B):
                pred_x, pred_y, pred_w, pred_h, pred_conf = pred_boxes[i, cell_y, cell_x, b]
                iou = calculate_iou(
                    (pred_x, pred_y, pred_w, pred_h), (x_center, y_center, box_w, box_h)
                )
                if iou > best_iou:
                    best_iou = iou
                    best_box = b

            # Compute localization loss for best box
            pred_box = pred_boxes[i, cell_y, cell_x, best_box]
            coord_loss += lambda_coord * (
                F.mse_loss(pred_box[0], x_center) + F.mse_loss(pred_box[1], y_center) +
                F.mse_loss(pred_box[2], box_w) + F.mse_loss(pred_box[3], box_h)
            )

            # Confidence loss
            conf_loss += F.mse_loss(pred_box[4], torch.tensor(float(best_iou), device=pred_box[4].device))

            # Classification loss (single class prediction per cell)
            target_class = target_labels[box_idx].float()  # Convert to float for consistency
            class_loss += F.cross_entropy(pred_classes[i, cell_y, cell_x].unsqueeze(0), target_class.unsqueeze(0).long())

        total_loss += coord_loss + conf_loss + class_loss

    return total_loss / batch_size

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IOU) between two bounding boxes.
    Each box is represented as (x, y, width, height).
    """
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    return inter_area / (box1_area + box2_area - inter_area)

# Integrate with the train_model function
def train_model(model, train_loader, val_loader, device, config):
    epochs = get_param(config, "epochs", 10)
    learning_rate = get_param(config, "learning_rate", 0.001)

    S = config["S"]
    B = config["B"]
    num_classes = config["num_classes"]

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc="Training", leave=False) as pbar:
            for images, targets in train_loader:
                images = torch.stack([image.to(device) for image in images]).to(device)
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

                optimizer.zero_grad()
                outputs = model(images)

                # Calculate loss using the compute_yolo_loss function
                losses = compute_yolo_loss(outputs, targets, S, B, num_classes)

                losses.backward()
                optimizer.step()

                running_loss += losses.item()
                pbar.update(1)
                pbar.set_postfix(loss=losses.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        
def pretrain_on_imagenet(model, config, device):
    # Retrieve parameters from config with defaults
    learning_rate = get_param(config, "pretrain_learning_rate", 0.001)
    batch_size = get_param(config, "pretrain_batch_size", 32)
    epochs = get_param(config, "pretrain_epochs", 10)
    
    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define ImageNet-like transforms for 224x224 resolution
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load a sample ImageNet dataset (replace with actual ImageNet dataset path in practice)
    imagenet_data = datasets.FakeData(transform=transform)  # Use actual ImageNet dataset in practice
    train_size = int(0.9 * len(imagenet_data))
    val_size = len(imagenet_data) - train_size
    train_data, val_data = random_split(imagenet_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):  # Outer progress bar for epochs
        running_loss = 0.0
        
        # Inner progress bar for training batches within the current epoch
        for inputs, labels in tqdm(train_loader, desc="Training Batches", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase with progress bar
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in tqdm(val_loader, desc="Validation Batches", leave=False):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
        
        model.train()
        
        # Print epoch summary after each epoch
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Training Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss / len(val_loader):.4f}")
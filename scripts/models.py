import torchvision
from scripts.utils import get_param
import torch.nn as nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torch.nn.init as init

def get_object_detection_model(config):
    num_classes = get_param(config, "num_classes", 37)  # Default to 37 classes

    # Load a pre-trained Faster R-CNN model with the weights parameter
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1  # Use COCO weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    
    # Modify the classifier head to the desired number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

class YOLOPretrain(nn.Module):
    def __init__(self):
        super(YOLOPretrain, self).__init__()

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Intermediate Layer between Layer 2 and Layer 3
        self.intermediate_layer = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 3: Repeat (1x1x256 -> 3x3x512) four times
        layer3_blocks = []
        for _ in range(4):
            layer3_blocks.extend([
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU()
            ])
        layer3_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(*layer3_blocks)

        # Layer 4 Part 1
        self.layer4_part1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Fully connected layer for pretraining
        self.pretrain_fc = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.intermediate_layer(x)
        x = self.layer3(x)
        x = self.layer4_part1(x)

        # Flatten and apply dynamically initialized fully connected layer
        x = x.view(x.size(0), -1)

        if self.pretrain_fc is None:
            # print(f"Initializing pretrain_fc with input size: {x.size(1)}")
            self.pretrain_fc = nn.Linear(x.size(1), 1000).to(x.device)

        x = self.pretrain_fc(x)
        return x
    
class YOLOTrain(nn.Module):
    def __init__(self, config):
        super(YOLOTrain, self).__init__()

        self.num_classes = config["num_classes"]
        self.B = config["B"]
        self.S = config["S"]
        output_size = self.S * self.S * (self.B * 5 + self.num_classes)  # Final output size

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Intermediate Layer between Layer 2 and Layer 3
        self.intermediate_layer = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 3: Repeat (1x1x256 -> 3x3x512) four times
        layer3_blocks = []
        for _ in range(4):
            layer3_blocks.extend([
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU()
            ])
        layer3_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(*layer3_blocks)

        # Layer 4 Part 1
        self.layer4_part1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Layer 4 Part 2 and Layer 5 for full mode
        self.layer4_part2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc1 = None
        self.output_layer = nn.Linear(4096, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.intermediate_layer(x)
        x = self.layer3(x)
        x = self.layer4_part1(x)
        x = self.layer4_part2(x)
        x = self.layer5(x)

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)

        if self.fc1 is None:
            print(f"Initializing fc1 with input size: {x.size(1)}")
            self.fc1 = nn.Linear(x.size(1), 4096).to(x.device)

        x = self.fc1(x)
        x = self.output_layer(x)  # Final layer to match (S * S * (B * 5 + C))

        # Reshape output to (batch_size, S, S, (B * 5 + C))
        x = x.view(-1, self.S, self.S, self.B * 5 + self.num_classes)

        return x
    
def transfer_weights(pretrain_model, train_model):
    # Copy weights for shared layers from pretrain_model to train_model
    train_model.layer1.load_state_dict(pretrain_model.layer1.state_dict())
    train_model.layer2.load_state_dict(pretrain_model.layer2.state_dict())
    train_model.intermediate_layer.load_state_dict(pretrain_model.intermediate_layer.state_dict())
    train_model.layer3.load_state_dict(pretrain_model.layer3.state_dict())
    train_model.layer4_part1.load_state_dict(pretrain_model.layer4_part1.state_dict())
    
    print("Weights transferred from YOLOPretrain to YOLOTrain.")

    # Initialize non-transferred layers with Xavier initialization
    def xavier_init(layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

    # Apply Xavier initialization to layers not transferred
    train_model.layer4_part2.apply(xavier_init)
    train_model.layer5.apply(xavier_init)

    # Initialize fully connected layers in the train model
    if train_model.fc1 is not None:
        xavier_init(train_model.fc1)
    xavier_init(train_model.output_layer)

    print("Non-transferred layers initialized with Xavier initialization.")
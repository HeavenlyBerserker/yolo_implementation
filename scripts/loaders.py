from scripts.utils import get_param
import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import shutil

def preprocess(input_root, output_root):
    """Preprocess the dataset by resizing images to a square 448x448 with padding and adjust bounding boxes."""
    
    # Set up directories
    image_input_dir = os.path.join(input_root, "images", "images")
    annotation_input_dir = os.path.join(input_root, "annotations", "annotations", "xmls")
    image_output_dir = os.path.join(output_root, "images")
    annotation_output_dir = os.path.join(output_root, "annotations", "xmls")
    
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(annotation_output_dir, exist_ok=True)

    # Process each image and its corresponding annotation
    for image_name in sorted(os.listdir(image_input_dir)):
        if not image_name.endswith(".jpg"):
            continue

        # Load image and its corresponding annotation
        img_path = os.path.join(image_input_dir, image_name)
        annotation_path = os.path.join(annotation_input_dir, image_name.replace(".jpg", ".xml"))

        # Skip if annotation doesn't exist
        if not os.path.exists(annotation_path):
            continue

        img = Image.open(img_path).convert("RGB")
        original_width, original_height = img.size

        # Parse the annotation file and collect bounding boxes
        boxes = []
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        # Resize and pad the image to 448x448 and adjust the bounding boxes
        img, boxes = resize_and_pad_image(img, boxes, original_width, original_height)

        # Save the preprocessed image
        img.save(os.path.join(image_output_dir, image_name))

        # Update and save the adjusted annotation
        for i, box in enumerate(boxes):
            bbox = root.findall("object")[i].find("bndbox")
            bbox.find("xmin").text = str(int(box[0]))
            bbox.find("ymin").text = str(int(box[1]))
            bbox.find("xmax").text = str(int(box[2]))
            bbox.find("ymax").text = str(int(box[3]))
        
        tree.write(os.path.join(annotation_output_dir, image_name.replace(".jpg", ".xml")))

    print(f"Preprocessed images and annotations have been saved to {output_root}")

def resize_and_pad_image(img, boxes, original_width, original_height):
    # Determine the scaling factor to resize the image to fit within 448x448 while maintaining aspect ratio
    scale = 448 / max(original_width, original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    img = img.resize((new_width, new_height), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
    
    # Calculate padding to make the image 448x448
    delta_width = 448 - new_width
    delta_height = 448 - new_height
    padding_left = delta_width // 2
    padding_right = delta_width - padding_left
    padding_top = delta_height // 2
    padding_bottom = delta_height - padding_top

    # Pad the image to make it 448x448
    img = ImageOps.expand(img, (padding_left, padding_top, padding_right, padding_bottom), fill=(0, 0, 0))

    # Adjust bounding boxes based on the scaling and padding
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + padding_left  # Adjust x coordinates
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + padding_top   # Adjust y coordinates

    return img, boxes.tolist()

def custom_collate_fn(batch):
    images, targets = zip(*batch)  # Separates images and targets
    images = torch.stack(images)   # Stack images into a single tensor (batch_size, 3, 448, 448)
    return images, targets

def get_pet_data_loaders(config):
    root_dir = get_param(config, "root_dir", "data/pets/preprocessed")
    batch_size = get_param(config, "batch_size", 2)

    # Define basic transforms for object detection (convert image to tensor)
    transforms = T.Compose([T.ToTensor()])

    # Create datasets for train, val, and test
    train_dataset = PetDataset(root=root_dir, image_set="train", transforms=transforms)
    val_dataset = PetDataset(root=root_dir, image_set="val", transforms=transforms)
    test_dataset = PetDataset(root=root_dir, image_set="test", transforms=transforms)

    # Use the custom collate function for object detection tasks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader

class PetDataset(Dataset):
    def __init__(self, root, image_set='train', transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_set = image_set
        self.image_dir = os.path.join(root, "images")
        self.annotation_dir = os.path.join(root, "annotations", "xmls")

        # Get image file names
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])

        # Filter images that have corresponding XML annotations
        self.images = [
            img for img in self.images
            if os.path.exists(os.path.join(self.annotation_dir, img.replace(".jpg", ".xml")))
        ]

        # Ensure we have images before proceeding
        if len(self.images) == 0:
            raise ValueError(f"No matching images and annotations found in directories {self.image_dir} and {self.annotation_dir}.")

        # Split data into train/val/test
        n = len(self.images)
        if image_set == 'train':
            self.images = self.images[:int(0.8 * n)]
        elif image_set == 'val':
            self.images = self.images[int(0.8 * n):int(0.9 * n)]
        elif image_set == 'test':
            self.images = self.images[int(0.9 * n):]

        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.image_set} split. Check dataset split logic.")

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        annotation_path = os.path.join(self.annotation_dir, self.images[idx].replace(".jpg", ".xml"))

        # Load the preprocessed image
        img = Image.open(img_path).convert("RGB")

        # Parse the XML annotation file
        boxes = []
        labels = []
        tree = ET.parse(annotation_path)
        for obj in tree.findall("object"):
            label = 1  # Assign a label; adjust if using multiple classes
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)
import torch
from tqdm import tqdm
from scripts.utils import get_param

def test_model(model, test_loader, device, config):
    S = config["S"]
    B = config["B"]
    num_classes = config["num_classes"]

    model.eval()
    model.to(device)

    print("Starting testing phase...")
    results = []

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Testing", leave=False) as pbar:
            for images, targets in test_loader:
                images = torch.stack([image.to(device) for image in images]).to(device)

                # Run the model to get predictions
                outputs = model(images)  # (batch_size, S, S, B * 5 + num_classes)

                # Process each image in the batch
                for i in range(images.size(0)):
                    output = outputs[i]
                    target = targets[i]
                    
                    # Reshape outputs for easier extraction
                    output = output.view(S, S, B * 5 + num_classes)
                    
                    pred_boxes = []
                    pred_scores = []
                    pred_labels = []

                    # Extract boxes, scores, and class predictions for each cell in the grid
                    for row in range(S):
                        for col in range(S):
                            cell_data = output[row, col]
                            
                            # Extract bounding boxes and confidence for each box
                            for b in range(B):
                                box_data = cell_data[b * 5:(b + 1) * 5]
                                x, y, w, h, confidence = box_data
                                score = confidence.item()

                                if score > 0.5:  # Threshold for confidence score
                                    # Convert cell-relative (x, y) to absolute coordinates
                                    abs_x = (col + x.item()) / S
                                    abs_y = (row + y.item()) / S
                                    abs_w = w.item()
                                    abs_h = h.item()

                                    pred_boxes.append([abs_x, abs_y, abs_w, abs_h])
                                    pred_scores.append(score)
                                    
                            # Extract predicted class
                            class_data = cell_data[B * 5:]
                            pred_class = torch.argmax(class_data).item()
                            pred_labels.append(pred_class)

                    # Collect ground truth
                    target_boxes = target['boxes'].cpu().numpy()
                    target_labels = target['labels'].cpu().numpy()

                    results.append({
                        "pred_boxes": pred_boxes,
                        "pred_labels": pred_labels,
                        "pred_scores": pred_scores,
                        "target_boxes": target_boxes,
                        "target_labels": target_labels,
                    })

                pbar.update(1)

    # # Basic report
    # for idx, result in enumerate(results):
    #     print(f"\nImage {idx + 1}")
    #     print(f"Predicted boxes: {len(result['pred_boxes'])}")
    #     print(f"Ground truth boxes: {len(result['target_boxes'])}")

    print("Testing completed.")

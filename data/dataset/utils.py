import os
import json
from PIL import Image


def add_samples(samples, data_list, root):
    """
    Add samples to the samples list.
    
    Args:
        samples: List to add samples to
        data_list: Path to data list file
        root: Root directory
    """
    with open(data_list, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                path, label = line.split(' ')
                samples.append((os.path.join(root, path), int(label)))


def create_annotation_file(data_path, splits, output_file):
    """
    Create annotation file for dataset.
    
    Args:
        data_path: Path to data directory
        splits: List of splits to include
        output_file: Output file path
    """
    samples = []
    
    # This is a placeholder implementation
    # In practice, this would scan the data directory and create annotations
    print(f"Creating annotation file: {output_file}")
    print(f"Data path: {data_path}")
    print(f"Splits: {splits}")
    
    # Create empty file for now
    with open(output_file, 'w') as f:
        pass


def get_transformation(mode, mean, std):
    """
    Get transformation for dataset.
    
    Args:
        mode: 'train' or 'val'
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        transform: Transformation pipeline
    """
    from torchvision import transforms
    
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # val
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform 
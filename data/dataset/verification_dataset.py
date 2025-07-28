import json
import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class VerificationDataset(Dataset):
    """
    Dataset for verification evaluation.
    
    Creates pairs of images for verification testing:
    - Positive pairs: Same person, different images
    - Negative pairs: Twin pairs (hard negatives)
    """
    
    def __init__(self, id_to_images_path, twin_pairs_path, data_root, mode='train', transform=None):
        """
        Args:
            id_to_images_path: Path to id_to_images.json
            twin_pairs_path: Path to twin pairs JSON
            data_root: Root directory containing face images
            mode: 'train' or 'test'
            transform: Image transformations
        """
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.loader = default_loader
        
        # Load dataset structure
        with open(id_to_images_path, 'r') as f:
            self.id_to_images = json.load(f)
        
        with open(twin_pairs_path, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Create verification pairs
        self.verification_pairs = self._create_verification_pairs()
        
        print(f"Created {len(self.verification_pairs)} verification pairs for {mode} set")
        positive_count = sum(1 for pair in self.verification_pairs if pair['label'] == 1)
        negative_count = sum(1 for pair in self.verification_pairs if pair['label'] == 0)
        print(f"Positive pairs: {positive_count}, Negative pairs: {negative_count}")
    
    def _create_verification_pairs(self):
        """Create verification pairs for evaluation."""
        verification_pairs = []
        
        # Create twin pair mapping
        twin_mapping = {}
        for person_id, twin_id in self.twin_pairs:
            twin_mapping[person_id] = twin_id
            twin_mapping[twin_id] = person_id
        
        # Generate positive pairs (same person)
        for person_id, images in self.id_to_images.items():
            if len(images) >= 2:
                # Create positive pairs from same person
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        verification_pairs.append({
                            'img1': images[i],
                            'img2': images[j],
                            'label': 1,  # Same person
                            'person_id': person_id
                        })
        
        # Generate negative pairs (twin pairs - hard negatives)
        for person_id, twin_id in self.twin_pairs:
            person_images = self.id_to_images.get(person_id, [])
            twin_images = self.id_to_images.get(twin_id, [])
            
            if len(person_images) > 0 and len(twin_images) > 0:
                # Create negative pairs between person and their twin
                for person_img in person_images:
                    for twin_img in twin_images:
                        verification_pairs.append({
                            'img1': person_img,
                            'img2': twin_img,
                            'label': 0,  # Different person (twin)
                            'person_id': person_id,
                            'twin_id': twin_id
                        })
        
        return verification_pairs
    
    def __len__(self):
        return len(self.verification_pairs)
    
    def __getitem__(self, idx):
        """Get a verification pair."""
        pair = self.verification_pairs[idx]
        
        # Load images
        img1_path = os.path.join(self.data_root, pair['img1'])
        img2_path = os.path.join(self.data_root, pair['img2'])
        
        img1 = self.loader(img1_path)
        img2 = self.loader(img2_path)
        
        # Apply transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # Create output dictionary
        output = {
            'img1': img1,
            'img2': img2,
            'label': pair['label']
        }
        
        # Add optional metadata
        if 'person_id' in pair:
            output['person_id'] = pair['person_id']
        if 'twin_id' in pair:
            output['twin_id'] = pair['twin_id']
        
        return output


def get_verification_dataset(params, mode='trainval_combined'):
    """
    Get verification dataset for evaluation.
    
    Args:
        params: Configuration parameters
        mode: 'trainval_combined' or 'test'
    
    Returns:
        VerificationDataset instance
    """
    # Define transforms (no augmentation, just normalization)
    mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    
    transform = transforms.Compose([
        transforms.Resize((params.crop_size, params.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    if mode == 'trainval_combined':
        # Training data
        id_to_images_path = 'id_to_images.json'
        twin_pairs_path = 'train_twin_id_pairs.json'
        
        if not os.path.exists(id_to_images_path):
            raise FileNotFoundError(f"id_to_images.json not found at {id_to_images_path}")
        if not os.path.exists(twin_pairs_path):
            raise FileNotFoundError(f"train_twin_id_pairs.json not found at {twin_pairs_path}")
        
        return VerificationDataset(
            id_to_images_path=id_to_images_path,
            twin_pairs_path=twin_pairs_path,
            data_root=params.data_path,
            mode='train',
            transform=transform
        )
    
    elif mode == 'test':
        # Test data
        id_to_images_path = 'id_to_images.json'
        twin_pairs_path = 'test_twin_id_pairs.json'
        
        if not os.path.exists(id_to_images_path):
            raise FileNotFoundError(f"id_to_images.json not found at {id_to_images_path}")
        if not os.path.exists(twin_pairs_path):
            raise FileNotFoundError(f"test_twin_id_pairs.json not found at {twin_pairs_path}")
        
        return VerificationDataset(
            id_to_images_path=id_to_images_path,
            twin_pairs_path=twin_pairs_path,
            data_root=params.data_path,
            mode='test',
            transform=transform
        )
    
    else:
        raise NotImplementedError(f"Mode {mode} not supported") 
import json
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class TwinFaceDataset(Dataset):
    """
    Twin Face Dataset for verification task using triplet loss.
    
    Generates triplets: (anchor, positive, negative)
    - anchor: random image from person A
    - positive: different image from person A (same person)
    - negative: random image from person A's twin (hard negative)
    """
    
    def __init__(self, id_to_images_path, twin_pairs_path, data_root, mode='train', transform=None, triplet_portion=1.0):
        """
        Args:
            id_to_images_path: Path to id_to_images.json
            twin_pairs_path: Path to train_twin_id_pairs.json or test_twin_id_pairs.json
            data_root: Root directory containing face images
            mode: 'train' or 'test'
            transform: Image transformations
            triplet_portion: Fraction of triplets to use (0.0-1.0). Ensures all twin pairs are represented.
        """
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.loader = default_loader
        self.triplet_portion = triplet_portion
        
        # Load dataset structure
        with open(id_to_images_path, 'r') as f:
            self.id_to_images = json.load(f)
        
        with open(twin_pairs_path, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Create twin pair mapping for easy lookup
        self.twin_mapping = {}
        for person_id, twin_id in self.twin_pairs:
            self.twin_mapping[person_id] = twin_id
            self.twin_mapping[twin_id] = person_id
        
        # Create list of valid person IDs (those with twins)
        self.valid_person_ids = list(self.twin_mapping.keys())
        
        # Create triplets for training
        self.triplets = self._create_triplets()
        
        print(f"Created {len(self.triplets)} triplets for {mode} set (using {triplet_portion*100:.1f}% of possible triplets)")
        print(f"Number of twin pairs: {len(self.twin_pairs)}")
        print(f"Number of unique persons: {len(self.valid_person_ids)}")
    
    def _create_triplets(self):
        """Create triplets for training/testing."""
        triplets = []
        
        for person_id in self.valid_person_ids:
            twin_id = self.twin_mapping[person_id]
            
            # Get images for current person and their twin
            person_images = self.id_to_images.get(person_id, [])
            twin_images = self.id_to_images.get(twin_id, [])
            
            # Skip if either person has less than 2 images
            if len(person_images) < 2 or len(twin_images) < 1:
                continue
            
            # Calculate number of triplets for this person based on triplet_portion
            max_triplets_per_person = min(len(person_images), 10)  # Original limit
            num_triplets_for_person = max(1, int(max_triplets_per_person * self.triplet_portion))
            
            # Create triplets for this person
            for _ in range(num_triplets_for_person):
                # Randomly select anchor and positive from same person
                anchor_img, positive_img = random.sample(person_images, 2)
                
                # Randomly select negative from twin
                negative_img = random.choice(twin_images)
                
                triplets.append({
                    'anchor': anchor_img,
                    'positive': positive_img,
                    'negative': negative_img,
                    'person_id': person_id,
                    'twin_id': twin_id
                })
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """Get a triplet."""
        triplet = self.triplets[idx]
        
        # Load images
        anchor_path = os.path.join(self.data_root, triplet['anchor'])
        positive_path = os.path.join(self.data_root, triplet['positive'])
        negative_path = os.path.join(self.data_root, triplet['negative'])
        
        anchor_img = self.loader(anchor_path)
        positive_img = self.loader(positive_path)
        negative_img = self.loader(negative_path)
        
        # Apply transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return {
            'anchor': anchor_img,
            'positive': positive_img,
            'negative': negative_img,
            'person_id': triplet['person_id'],
            'twin_id': triplet['twin_id']
        }


def get_twin_faces(params, mode='trainval_combined'):
    """
    Get twin face dataset.
    
    Args:
        params: Configuration parameters
        mode: 'trainval_combined' or 'test'
    
    Returns:
        TwinFaceDataset instance
    """
    # Set number of classes to 2 (same person vs twin)
    params.class_num = 2
    
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
        
        # Get triplet_portion from params, default to 1.0 if not specified
        triplet_portion = getattr(params, 'triplet_portion', 1.0)
        
        return TwinFaceDataset(
            id_to_images_path=id_to_images_path,
            twin_pairs_path=twin_pairs_path,
            data_root=params.data_path,
            mode='train',
            transform=transform,
            triplet_portion=triplet_portion
        )
    
    elif mode == 'test':
        # Test data
        id_to_images_path = 'id_to_images.json'
        twin_pairs_path = 'test_twin_id_pairs.json'
        
        if not os.path.exists(id_to_images_path):
            raise FileNotFoundError(f"id_to_images.json not found at {id_to_images_path}")
        if not os.path.exists(twin_pairs_path):
            raise FileNotFoundError(f"test_twin_id_pairs.json not found at {twin_pairs_path}")
        
        # For test mode, always use full dataset (triplet_portion=1.0)
        return TwinFaceDataset(
            id_to_images_path=id_to_images_path,
            twin_pairs_path=twin_pairs_path,
            data_root=params.data_path,
            mode='test',
            transform=transform,
            triplet_portion=1.0  # Always use full test set
        )
    
    else:
        raise NotImplementedError(f"Mode {mode} not supported")


def create_verification_pairs(id_to_images_path, twin_pairs_path, mode='train'):
    """
    Create verification pairs for evaluation.
    
    Args:
        id_to_images_path: Path to id_to_images.json
        twin_pairs_path: Path to twin pairs JSON
        mode: 'train' or 'test'
    
    Returns:
        List of verification pairs with labels
    """
    with open(id_to_images_path, 'r') as f:
        id_to_images = json.load(f)
    
    with open(twin_pairs_path, 'r') as f:
        twin_pairs = json.load(f)
    
    verification_pairs = []
    
    # Create twin pair mapping
    twin_mapping = {}
    for person_id, twin_id in twin_pairs:
        twin_mapping[person_id] = twin_id
        twin_mapping[twin_id] = person_id
    
    # Generate positive pairs (same person)
    for person_id, images in id_to_images.items():
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
    for person_id, twin_id in twin_pairs:
        person_images = id_to_images.get(person_id, [])
        twin_images = id_to_images.get(twin_id, [])
        
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
    
    print(f"Created {len(verification_pairs)} verification pairs for {mode} set")
    positive_count = sum(1 for pair in verification_pairs if pair['label'] == 1)
    negative_count = sum(1 for pair in verification_pairs if pair['label'] == 0)
    print(f"Positive pairs: {positive_count}, Negative pairs: {negative_count}")
    
    return verification_pairs 
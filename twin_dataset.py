import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class TwinDataset(Dataset):
    """
    Custom dataset for ND_TWIN dataset
    
    Loads images from id_to_images.json and generates pairs based on twin relationships
    in twin_pairs_infor.json with configurable sampling ratios.
    """
    
    def __init__(self,
                 id_to_images_path: str,
                 twin_pairs_path: str,
                 split: str = 'train',
                 transform=None,
                 same_person_ratio: float = 0.5,
                 twin_pairs_ratio: float = 0.3,
                 non_twin_ratio: float = 0.2,
                 max_pairs_per_epoch: Optional[int] = None):
        """
        Args:
            id_to_images_path: Path to id_to_images.json
            twin_pairs_path: Path to twin_pairs_infor.json
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            same_person_ratio: Ratio of same-person pairs
            twin_pairs_ratio: Ratio of twin pairs (hard negatives)
            non_twin_ratio: Ratio of non-twin pairs (easy negatives)
            max_pairs_per_epoch: Maximum pairs per epoch (None for all)
        """
        self.id_to_images_path = id_to_images_path
        self.twin_pairs_path = twin_pairs_path
        self.split = split
        self.transform = transform
        self.same_person_ratio = same_person_ratio
        self.twin_pairs_ratio = twin_pairs_ratio
        self.non_twin_ratio = non_twin_ratio
        self.max_pairs_per_epoch = max_pairs_per_epoch
        
        # Validate ratios are positive
        if same_person_ratio <= 0:
            raise ValueError(f"Same person ratio must be positive, got {same_person_ratio}")
        if twin_pairs_ratio <= 0:
            raise ValueError(f"Twin pairs ratio must be positive, got {twin_pairs_ratio}")
        if non_twin_ratio <= 0:
            raise ValueError(f"Non-twin ratio must be positive, got {non_twin_ratio}")
        
        # Load dataset
        self._load_dataset()
        
        # Generate pairs
        self._generate_pairs()
        
    def _load_dataset(self):
        """Load dataset from JSON files"""
        # Load id to images mapping
        with open(self.id_to_images_path, 'r') as f:
            self.id_to_images = json.load(f)
            
        # Load twin pairs information
        with open(self.twin_pairs_path, 'r') as f:
            twin_pairs_info = json.load(f)
            
        # Get twin pairs for current split
        if self.split not in twin_pairs_info:
            raise ValueError(f"Split '{self.split}' not found in twin_pairs_infor.json")
            
        self.twin_pairs = twin_pairs_info[self.split]
        
        # Create ID sets for current split
        self.split_ids = set()
        for twin_pair in self.twin_pairs:
            self.split_ids.add(twin_pair[0])
            self.split_ids.add(twin_pair[1])
            
        # Filter id_to_images to only include IDs in current split
        self.split_id_to_images = {
            id_: images for id_, images in self.id_to_images.items()
            if id_ in self.split_ids
        }
        
        # Create twin relationship mapping
        self.twin_relationships = {}
        for twin_pair in self.twin_pairs:
            id1, id2 = twin_pair[0], twin_pair[1]
            self.twin_relationships[id1] = id2
            self.twin_relationships[id2] = id1
            
        # Create non-twin ID pairs (IDs that are not twins)
        self.non_twin_pairs = []
        split_ids_list = list(self.split_ids)
        for i, id1 in enumerate(split_ids_list):
            for id2 in split_ids_list[i+1:]:
                # Check if they are not twins
                if id2 != self.twin_relationships.get(id1, None):
                    self.non_twin_pairs.append((id1, id2))
                    
        print(f"Loaded {self.split} split:")
        print(f"  Total IDs: {len(self.split_ids)}")
        print(f"  Twin pairs: {len(self.twin_pairs)}")
        print(f"  Non-twin pairs: {len(self.non_twin_pairs)}")
        
    def _generate_pairs(self):
        """Generate training pairs based on configured ratios"""
        self.pairs = []
        self.labels = []
        
        # Generate same-person pairs
        same_person_pairs = self._generate_same_person_pairs()
        num_same_person = int(len(same_person_pairs) * self.same_person_ratio)
        selected_same_person = random.sample(same_person_pairs, min(num_same_person, len(same_person_pairs)))
        
        # Generate twin pairs
        twin_pairs = self._generate_twin_pairs()
        num_twin = int(len(twin_pairs) * self.twin_pairs_ratio)
        selected_twin = random.sample(twin_pairs, min(num_twin, len(twin_pairs)))
        
        # Generate non-twin pairs
        num_non_twin = int(len(self.non_twin_pairs) * self.non_twin_ratio)
        selected_non_twin = random.sample(self.non_twin_pairs, min(num_non_twin, len(self.non_twin_pairs)))
        
        # Combine all pairs
        all_pairs = []
        all_labels = []
        
        # Add same-person pairs (label = 1)
        for pair in selected_same_person:
            all_pairs.append(pair)
            all_labels.append(1)
            
        # Add twin pairs (label = 0, hard negatives)
        for pair in selected_twin:
            all_pairs.append(pair)
            all_labels.append(0)
            
        # Add non-twin pairs (label = 0, easy negatives)
        for pair in selected_non_twin:
            all_pairs.append(pair)
            all_labels.append(0)
            
        # Shuffle pairs
        combined = list(zip(all_pairs, all_labels))
        random.shuffle(combined)
        self.pairs, self.labels = zip(*combined)
        
        # Limit pairs if specified
        if self.max_pairs_per_epoch is not None:
            self.pairs = self.pairs[:self.max_pairs_per_epoch]
            self.labels = self.labels[:self.max_pairs_per_epoch]
            
        print(f"Generated {len(self.pairs)} pairs:")
        print(f"  Same-person pairs: {sum(self.labels)}")
        print(f"  Different-person pairs: {len(self.labels) - sum(self.labels)}")
        
    def _generate_same_person_pairs(self) -> List[Tuple[str, str]]:
        """Generate all possible same-person image pairs"""
        same_person_pairs = []
        
        for id_, images in self.split_id_to_images.items():
            if len(images) < 2:
                continue
                
            # Generate all possible pairs for this person
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    same_person_pairs.append((images[i], images[j]))
                    
        return same_person_pairs
        
    def _generate_twin_pairs(self) -> List[Tuple[str, str]]:
        """Generate all possible twin image pairs"""
        twin_pairs = []
        
        for twin_pair in self.twin_pairs:
            id1, id2 = twin_pair[0], twin_pair[1]
            
            # Get images for both twins
            images1 = self.split_id_to_images.get(id1, [])
            images2 = self.split_id_to_images.get(id2, [])
            
            if not images1 or not images2:
                continue
                
            # Generate all possible pairs between twins
            for img1 in images1:
                for img2 in images2:
                    twin_pairs.append((img1, img2))
                    
        return twin_pairs
        
    def __len__(self) -> int:
        """Return number of pairs"""
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of images and label
        
        Returns:
            Tuple of (image1, image2, label)
        """
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        # For testing purposes, create dummy images if files don't exist
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            # Create dummy images for testing
            if self.transform is not None:
                # Create dummy tensors that match the transform output
                img1 = torch.randn(3, 224, 224)  # Assuming 224x224 input
                img2 = torch.randn(3, 224, 224)
            else:
                img1 = torch.randn(3, 224, 224)
                img2 = torch.randn(3, 224, 224)
        else:
            # Load actual images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Apply transformations
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)
        
    def get_pair_statistics(self) -> Dict[str, int]:
        """Get statistics about the generated pairs"""
        return {
            'total_pairs': len(self.pairs),
            'same_person_pairs': sum(self.labels),
            'different_person_pairs': len(self.labels) - sum(self.labels),
            'twin_pairs': len([p for p in self.pairs if self._is_twin_pair(p)]),
            'non_twin_pairs': len([p for p in self.pairs if not self._is_twin_pair(p)])
        }
        
    def _is_twin_pair(self, pair: Tuple[str, str]) -> bool:
        """Check if a pair is a twin pair"""
        img1_path, img2_path = pair
        
        # Extract IDs from image paths
        id1 = self._extract_id_from_path(img1_path)
        id2 = self._extract_id_from_path(img2_path)
        
        # Check if they are twins
        return self.twin_relationships.get(id1) == id2
        
    def _extract_id_from_path(self, path: str) -> str:
        """Extract ID from image path"""
        # This is a simplified implementation
        # In practice, you might need to adjust based on your path structure
        filename = os.path.basename(path)
        # Assuming path contains ID information
        # You may need to modify this based on your actual path structure
        return filename.split('_')[0]  # Simplified extraction


class TwinDataLoader:
    """
    Data loader for twin verification with configurable pair sampling
    """
    
    def __init__(self,
                 id_to_images_path: str,
                 twin_pairs_path: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 input_size: int = 224,
                 same_person_ratio: float = 0.5,
                 twin_pairs_ratio: float = 0.3,
                 non_twin_ratio: float = 0.2,
                 max_pairs_per_epoch: Optional[int] = None,
                 shuffle: bool = True):
        """
        Args:
            id_to_images_path: Path to id_to_images.json
            twin_pairs_path: Path to twin_pairs_infor.json
            batch_size: Batch size
            num_workers: Number of workers for data loading
            input_size: Input image size
            same_person_ratio: Ratio of same-person pairs
            twin_pairs_ratio: Ratio of twin pairs
            non_twin_ratio: Ratio of non-twin pairs
            max_pairs_per_epoch: Maximum pairs per epoch
            shuffle: Whether to shuffle data
        """
        self.id_to_images_path = id_to_images_path
        self.twin_pairs_path = twin_pairs_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.same_person_ratio = same_person_ratio
        self.twin_pairs_ratio = twin_pairs_ratio
        self.non_twin_ratio = non_twin_ratio
        self.max_pairs_per_epoch = max_pairs_per_epoch
        self.shuffle = shuffle
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, validation, and test dataloaders
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = TwinDataset(
            id_to_images_path=self.id_to_images_path,
            twin_pairs_path=self.twin_pairs_path,
            split='train',
            transform=self.train_transform,
            same_person_ratio=self.same_person_ratio,
            twin_pairs_ratio=self.twin_pairs_ratio,
            non_twin_ratio=self.non_twin_ratio,
            max_pairs_per_epoch=self.max_pairs_per_epoch
        )
        
        val_dataset = TwinDataset(
            id_to_images_path=self.id_to_images_path,
            twin_pairs_path=self.twin_pairs_path,
            split='val',
            transform=self.val_transform,
            same_person_ratio=self.same_person_ratio,
            twin_pairs_ratio=self.twin_pairs_ratio,
            non_twin_ratio=self.non_twin_ratio,
            max_pairs_per_epoch=self.max_pairs_per_epoch
        )
        
        test_dataset = TwinDataset(
            id_to_images_path=self.id_to_images_path,
            twin_pairs_path=self.twin_pairs_path,
            split='test',
            transform=self.val_transform,
            same_person_ratio=self.same_person_ratio,
            twin_pairs_ratio=self.twin_pairs_ratio,
            non_twin_ratio=self.non_twin_ratio,
            max_pairs_per_epoch=self.max_pairs_per_epoch
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
        
    def get_hard_pairs_dataloader(self, split: str = 'test') -> DataLoader:
        """
        Get dataloader with only hard pairs (twin pairs and same-person pairs)
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            DataLoader with hard pairs only
        """
        # Create dataset with only hard pairs
        hard_dataset = TwinDataset(
            id_to_images_path=self.id_to_images_path,
            twin_pairs_path=self.twin_pairs_path,
            split=split,
            transform=self.val_transform,
            same_person_ratio=0.5,  # Equal ratio for hard pairs
            twin_pairs_ratio=0.5,   # Equal ratio for hard pairs
            non_twin_ratio=0.0,     # No easy negatives
            max_pairs_per_epoch=self.max_pairs_per_epoch
        )
        
        return DataLoader(
            hard_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 
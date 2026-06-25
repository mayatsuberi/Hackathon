import torch
from torchvision import transforms

def get_robust_transforms(image_size: int) -> transforms.Compose:
    """
    Creates a robust data augmentation pipeline for training.
    
    This pipeline combines spatial transformations, color jitters, and 
    random erasing to prevent overfitting. It forces the model to learn 
    invariant features of the subjects rather than relying on background, 
    lighting, or specific image orientations.
    
    Args:
        image_size (int): The target size for the images (e.g., 224).
        
    Returns:
        transforms.Compose: The complete transformation pipeline.
    """
    # ImageNet standard normalization values
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    return transforms.Compose([
    transforms.RandomResizedCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.5)
])
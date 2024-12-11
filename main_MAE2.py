import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from functools import partial
from tqdm import tqdm
import wandb  # Optional: for logging
import numpy as np
from MAE_2_0 import MaskedAutoencoderViT

def create_dataloader(data_path, transform, batch_size=256, num_workers=4, shuffle=True):
    """
    Create DataLoader for training or testing.
    
    Args:
        data_path (str): Path to image dataset
        transform (callable): Image transformations
        batch_size (int): Batch size for data loading
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the data
    
    Returns:
        torch.utils.data.DataLoader: Configured DataLoader
    """
    dataset = datasets.ImageFolder(
        root=data_path, 
        transform=transform
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )

def evaluate(model, test_loader, device, config):
    """
    Evaluate the model on test data.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
        config (dict): Configuration dictionary
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    # Lists to store detailed metrics
    batch_losses = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluation")
        for batch in progress_bar:
            images, _ = batch  # We only need images for self-supervised learning
            images = images.to(device)
            
            # Forward pass with random masking
            loss, pred, mask = model(images, mask_ratio=config['mask_ratio'])

            # Ensure loss is reduced across GPUs
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()  # Reduce loss across devices
            
            total_loss += loss.item()
            batch_losses.append(loss.item())
            total_batches += 1
            
            progress_bar.set_postfix({'test_loss': loss.item()})
    
    # Compute evaluation metrics
    avg_loss = total_loss / total_batches
    loss_std = np.std(batch_losses) if len(batch_losses) > 1 else 0
    
    metrics = {
        'avg_test_loss': avg_loss,
        'test_loss_std': loss_std,
        'total_batches': total_batches
    }
    
    # Optional: Logging with wandb
    if config.get('use_wandb', False):
        wandb.log(metrics)
    
    return metrics

def train(model, train_loader, optimizer, device, config):
    """
    Training loop for a single epoch.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (torch.optim): Optimizer
        device (torch.device): Device to run training on
        config (dict): Training configuration
    
    Returns:
        dict: Training results for the epoch
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        images, _ = batch  # We only need images for self-supervised learning
        images = images.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with random masking
        loss, pred, mask = model(images, mask_ratio=config['mask_ratio'])

        # Ensure loss is reduced across GPUs
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()  # Reduce loss across devices
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
        progress_bar.set_postfix({'train_loss': loss.item()})
    
    # Compute training metrics
    avg_train_loss = total_loss / total_batches
    
    return {
        'avg_train_loss': avg_train_loss,
        'total_batches': total_batches
    }
    
def main():
    # Configuration
    config = {
        'train_data_path': '/mnt/d/OneDrive - Rowan University/RA/Fall 24/DenoMAE_2.0/sample_generation/data/unlabeled/train/noiseLessImg/',  # Update with your train dataset path
        'test_data_path': '/mnt/d/OneDrive - Rowan University/RA/Fall 24/DenoMAE_2.0/sample_generation/data/unlabeled/test/noiseLessImg/',    # Update with your test dataset path
        'batch_size': 256,
        'test_batch_size': 10,
        'num_epochs': 100,
        'learning_rate': 1.5e-4,
        'weight_decay': 0.05,
        'mask_ratio': 0.75,
        'use_wandb': True  # Set to True if you want to use wandb logging
    }
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optional: Initialize wandb
    if config['use_wandb']:
        wandb.init(project='mae-pretraining', config=config)
    
    # Define data augmentation and preprocessing transforms.
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create data loaders
    train_loader = create_dataloader(
        config['train_data_path'], 
        transform, 
        batch_size=config['batch_size']
    )
    test_loader = create_dataloader(
        config['test_data_path'], 
        transform, 
        batch_size=config['test_batch_size'], 
        shuffle=False
    )
    
    # Model initialization
    base_model = MaskedAutoencoderViT(
        img_size=224, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12,
        decoder_embed_dim=512, 
        decoder_depth=8, 
        decoder_num_heads=16,
        mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    # Multi-GPU support with nn.DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(base_model).to(device)
    else:
        model = base_model.to(device)
    
    # Optimizer setup
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs'], 
        eta_min=1e-6
    )
    
    # Training loop
    best_test_loss = float('inf')
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_results = train(
            model, 
            train_loader,
            optimizer, 
            device, 
            config
        )
        
        # Evaluation
        test_metrics = evaluate(
            model, 
            test_loader,
            device, 
            config
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        # Model checkpointing based on test loss
        current_test_loss = test_metrics['avg_test_loss']
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            checkpoint_path = 'models/mae_best_model.pth'
            
            # Handle checkpointing for DataParallel model
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_loss': best_test_loss
            }, checkpoint_path)
            print(f"Best model saved: {checkpoint_path}")
        
        # Print epoch summary
        print(f"Epoch Summary:")
        print(f"Train Loss: {train_results['avg_train_loss']:.4f}")
        print(f"Test Loss: {test_metrics['avg_test_loss']:.4f}")
        
        # Optional: Logging with wandb
        if config.get('use_wandb', False):
            wandb.log({
                **train_results,
                **test_metrics,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
    
    # Final model save
    # Handle saving for DataParallel model
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), 'models/mae_final_model.pth')
    else:
        torch.save(model.state_dict(), 'models/mae_final_model.pth')
    
    # Wandb finish
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()
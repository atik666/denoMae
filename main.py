import torch
from MultiMAE import MultiMAE
import argparse
from datagen import DenoMAEDataGenerator
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os

train_path = '/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/unlabeled/train/'
test_path = '/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/unlabeled/test/'
config = {'train_noisy_image_path' : train_path+'noisyImg/',
        'train_noiseless_image_path' : train_path+'noiseLessImg/',
        'train_noisy_signal_path' : train_path+'noisySignal/',
        'train_noiseless_signal_path': train_path+'noiselessSignal/',
        'train_noise_path' : train_path+'noise/',
        'test_noisy_image_path' : test_path+'noisyImg/',
        'test_noiseless_image_path' : test_path+'noiseLessImg/',
        'test_noisy_signal_path' : test_path+'noisySignal/',
        'test_noiseless_signal_path': test_path+'noiselessSignal/',
        'test_noise_path' : test_path+'noise/',
        'batch_size' : 16,
        'image_size' : (224, 224),
        }

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


# Create dataset and data loader
train_dataset = DenoMAEDataGenerator(noisy_image_path=config['train_noisy_image_path'], noiseless_img_path=config['train_noiseless_image_path'], 
                               noisy_signal_path=config['train_noisy_signal_path'], noiseless_signal_path=config['train_noiseless_signal_path'],   
                            noise_path = config['train_noise_path'], image_size=config['image_size'], transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

test_dataset = DenoMAEDataGenerator(noisy_image_path=config['test_noisy_image_path'], noiseless_img_path=config['test_noiseless_image_path'], 
                               noisy_signal_path=config['test_noisy_signal_path'], noiseless_signal_path=config['test_noiseless_signal_path'],   
                            noise_path = config['test_noise_path'], image_size=config['image_size'], transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

# Hyperparameters
img_size = 224
patch_size = 16
in_chans = 3
embed_dim = 768
encoder_depth = 12
decoder_depth = 4
num_heads = 12
batch_size = 32
num_epochs = 50
learning_rate = 1e-4

# TensorBoard setup
writer = SummaryWriter('runs/multimae')

# Initialize the model
model = MultiMAE(img_size, patch_size, in_chans, embed_dim, encoder_depth, decoder_depth, num_heads)

if os.path.exists('models/multimae_model.pth'):
    model.load_state_dict(torch.load('models/multimae_model.pth'))
    print("Model loaded successfully!")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (noisy_img, noisy_signal, noiseless_img, noiseless_signal, noise_data) in enumerate(train_dataloader):
        noisy_img, noisy_signal, noiseless_img, noiseless_signal, noise_data = noisy_img.to(device), noisy_signal.to(device), \
                                                                                noiseless_img.to(device), noiseless_signal.to(device), \
                                                                                noise_data.to(device) 

        # Forward pass
        rec1, rec2, rec3, mask1, mask2, mask3 = model(noisy_img, noisy_signal, noise_data)
        
        # Reshape masks to match image dimensions
        mask1 = mask1.view(mask1.shape[0], 1, img_size // patch_size, img_size // patch_size)
        mask2 = mask2.view(mask2.shape[0], 1, img_size // patch_size, img_size // patch_size)
        mask3 = mask3.view(mask3.shape[0], 1, img_size // patch_size, img_size // patch_size)
        mask1 = mask1.repeat(1, 3, patch_size, patch_size)
        mask2 = mask2.repeat(1, 3, patch_size, patch_size)
        mask3 = mask3.repeat(1, 3, patch_size, patch_size)
        
        # Compute loss only on masked regions
        loss1 = criterion(rec1 * (1 - mask1), noiseless_img * (1 - mask1))
        loss2 = criterion(rec2 * (1 - mask2), noiseless_signal * (1 - mask2))
        loss3 = criterion(rec3 * (1 - mask3), noise_data * (1 - mask3))
        loss = loss1 + loss2 + loss3
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    writer.add_scalar('Average Training Loss', avg_loss, epoch)

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_idx, (noisy_img, noisy_signal, noiseless_img, noiseless_signal, noise_data) in enumerate(test_dataloader):
            noisy_img, noisy_signal, noiseless_img, noiseless_signal, noise_data = noisy_img.to(device), noisy_signal.to(device), \
                                                                                   noiseless_img.to(device), noiseless_signal.to(device), \
                                                                                   noise_data.to(device)
            
            rec1, rec2, rec3, mask1, mask2, mask3 = model(noisy_img, noisy_signal, noise_data)
            
            # Reshape masks to match image dimensions
            mask1 = mask1.view(mask1.shape[0], 1, img_size // patch_size, img_size // patch_size)
            mask2 = mask2.view(mask2.shape[0], 1, img_size // patch_size, img_size // patch_size)
            mask3 = mask3.view(mask3.shape[0], 1, img_size // patch_size, img_size // patch_size)
            mask1 = mask1.repeat(1, 3, patch_size, patch_size)
            mask2 = mask2.repeat(1, 3, patch_size, patch_size)
            mask3 = mask3.repeat(1, 3, patch_size, patch_size)
            
            # Compute loss only on masked regions
            loss1 = criterion(rec1 * (1 - mask1), noiseless_img * (1 - mask1))
            loss2 = criterion(rec2 * (1 - mask2), noiseless_signal * (1 - mask2))
            loss3 = criterion(rec3 * (1 - mask3), noise_data * (1 - mask3))
            loss = loss1 + loss2 + loss3
            
            test_loss += loss.item()
        
            # Log input data and masks to TensorBoard
            grid = make_grid(torch.cat([
                noisy_img,
                rec1,
                noiseless_img
            ], dim=0), nrow=noisy_img.size(0))

            writer.add_image(f"Epoch {epoch}, Batch {batch_idx}", grid, epoch * len(test_dataloader) + batch_idx)

        avg_test_loss = test_loss / len(test_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")
        writer.add_scalar('Test Loss', avg_test_loss, epoch)

# Save the trained model
torch.save(model.state_dict(), 'models/multimae_model.pth')

print("Training completed!")

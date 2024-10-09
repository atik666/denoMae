import torch
import torch.nn as nn
from MultiMAE import MultiMAE
import argparse
from datagen import DenoMAEDataGenerator
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os

def parse_args():
    parser = argparse.ArgumentParser(description="MultiMAE Training Script")
    parser.add_argument("--train_path", type=str, default='/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/unlabeled/train/',
                        help="Path to training data")
    parser.add_argument("--test_path", type=str, default='/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/unlabeled/test/',
                        help="Path to test data")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224), help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--in_chans", type=int, default=3, help="Number of input channels")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=12, help="Depth of the encoder")
    parser.add_argument("--decoder_depth", type=int, default=4, help="Depth of the decoder")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_dir", type=str, default='runs/multimae', help="Directory for TensorBoard logs")
    parser.add_argument("--model_dir", type=str, default='models', help="Directory to save models")
    parser.add_argument("--model_name", type=str, default='multimae_model.pth', help="Name of the model file")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for data loader")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)

    config = {
        'train_noisy_image_path': os.path.join(args.train_path, 'noisyImg/'),
        'train_noiseless_image_path': os.path.join(args.train_path, 'noiseLessImg/'),
        'train_noisy_signal_path': os.path.join(args.train_path, 'noisySignal/'),
        'train_noiseless_signal_path': os.path.join(args.train_path, 'noiselessSignal/'),
        'train_noise_path': os.path.join(args.train_path, 'noise/'),
        'test_noisy_image_path': os.path.join(args.test_path, 'noisyImg/'),
        'test_noiseless_image_path': os.path.join(args.test_path, 'noiseLessImg/'),
        'test_noisy_signal_path': os.path.join(args.test_path, 'noisySignal/'),
        'test_noiseless_signal_path': os.path.join(args.test_path, 'noiselessSignal/'),
        'test_noise_path': os.path.join(args.test_path, 'noise/'),
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'patch_size': args.patch_size,
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and data loader
    train_dataset = DenoMAEDataGenerator(noisy_image_path=config['train_noisy_image_path'], noiseless_img_path=config['train_noiseless_image_path'], 
                                   noisy_signal_path=config['train_noisy_signal_path'], noiseless_signal_path=config['train_noiseless_signal_path'],   
                                noise_path=config['train_noise_path'], image_size=config['image_size'], transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=args.n_workers, pin_memory=True)

    test_dataset = DenoMAEDataGenerator(noisy_image_path=config['test_noisy_image_path'], noiseless_img_path=config['test_noiseless_image_path'], 
                                   noisy_signal_path=config['test_noisy_signal_path'], noiseless_signal_path=config['test_noiseless_signal_path'],   
                                noise_path=config['test_noise_path'], image_size=config['image_size'], transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=args.n_workers, pin_memory=True)

    # TensorBoard setup
    writer = SummaryWriter(args.log_dir)

    # Initialize the model
    model = MultiMAE(args.image_size[0], args.patch_size, args.in_chans, args.embed_dim, args.encoder_depth, args.decoder_depth, args.num_heads)

    # Check if CUDA is available and set up DataParallel
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (noisy_img, noisy_signal, noiseless_img, noiseless_signal, noise_data) in enumerate(train_dataloader):
            noisy_img, noisy_signal, noiseless_img, noiseless_signal, noise_data = noisy_img.to(device), noisy_signal.to(device), \
                                                                                    noiseless_img.to(device), noiseless_signal.to(device), \
                                                                                    noise_data.to(device) 

            # Forward pass
            rec1, rec2, rec3, mask1, mask2, mask3 = model(noisy_img, noisy_signal, noise_data)
            
            # Reshape masks to match image dimensions
            mask1 = mask1.view(mask1.shape[0], 1, args.image_size[0] // args.patch_size, args.image_size[1] // args.patch_size)
            mask2 = mask2.view(mask2.shape[0], 1, args.image_size[0] // args.patch_size, args.image_size[1] // args.patch_size)
            mask3 = mask3.view(mask3.shape[0], 1, args.image_size[0] // args.patch_size, args.image_size[1] // args.patch_size)
            mask1 = mask1.repeat(1, 3, args.patch_size, args.patch_size)
            mask2 = mask2.repeat(1, 3, args.patch_size, args.patch_size)
            mask3 = mask3.repeat(1, 3, args.patch_size, args.patch_size)
            
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
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}")
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
                mask1 = mask1.view(mask1.shape[0], 1, args.image_size[0] // args.patch_size, args.image_size[1] // args.patch_size)
                mask2 = mask2.view(mask2.shape[0], 1, args.image_size[0] // args.patch_size, args.image_size[1] // args.patch_size)
                mask3 = mask3.view(mask3.shape[0], 1, args.image_size[0] // args.patch_size, args.image_size[1] // args.patch_size)
                mask1 = mask1.repeat(1, 3, args.patch_size, args.patch_size)
                mask2 = mask2.repeat(1, 3, args.patch_size, args.patch_size)
                mask3 = mask3.repeat(1, 3, args.patch_size, args.patch_size)
                
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
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Test Loss: {avg_test_loss:.4f}")
            writer.add_scalar('Test Loss', avg_test_loss, epoch)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.module.state_dict(), os.path.join(args.model_dir, args.model_name))
            print(f"Model saved at epoch {epoch+1}")

    # Save the final model with a unique name
    final_model_path = os.path.join(args.model_dir, f'multimae_model_final_epoch{args.num_epochs}.pth')
    torch.save(model.module.state_dict(), final_model_path)
    print(f"Final model saved at: {final_model_path}")

    print("Training completed!")

if __name__ == "__main__":
    main()   

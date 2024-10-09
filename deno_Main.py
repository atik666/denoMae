import torch
import torch.nn as nn
from DenoMAE import DenoMAE
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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224), help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--in_chans", type=int, default=3, help="Number of input channels")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=12, help="Depth of the encoder")
    parser.add_argument("--decoder_depth", type=int, default=4, help="Depth of the decoder")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_dir", type=str, default='runs/multimae', help="Directory for TensorBoard logs")
    parser.add_argument("--model_dir", type=str, default='models', help="Directory to save models")
    parser.add_argument("--num_modality", type=int, default=1, help="Number of modalities")
    parser.add_argument("--model_name", type=str, default='multimae_model_dynamic_', help="Name of the model file")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--load", action="store_true", help="Load a previously saved model")
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
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=args.n_workers, pin_memory=True)

    # TensorBoard setup
    writer = SummaryWriter(args.log_dir)

    # Initialize the model
    model = DenoMAE(args.num_modality, args.image_size[0], args.patch_size, args.in_chans, args.embed_dim, args.encoder_depth, args.decoder_depth, args.num_heads)

    # Load model if a checkpoint exists
    if args.load and os.path.exists(args.model_dir):
        print("Loading model from checkpoint...")
        checkpoint = torch.load(args.model_dir+'/'+args.model_name+str(args.num_modality)+'.pth')
        model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found, initializing model.")

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

    # Initialize a variable to store the minimum test loss
    best_test_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):

            # Move all inputs and targets to the device
            inputs = [x.to(device) for x in inputs]
            targets = [x.to(device) for x in targets]

            # Forward pass
            reconstructions, masks = model(inputs)

            # Compute loss for each modality
            losses = []
            for rec, target, mask in zip(reconstructions, targets, masks):
                # Reshape mask to match input dimensions
                mask_reshaped = mask.view(mask.shape[0], 1, args.image_size[0] // args.patch_size, args.image_size[1] // args.patch_size)
                mask_reshaped = mask_reshaped.repeat(1, target.size(1), args.patch_size, args.patch_size)
                
                # Compute loss only on masked regions
                loss = criterion(rec * (1 - mask_reshaped), target * (1 - mask_reshaped))
                losses.append(loss)

            # Combine losses
            loss = sum(losses)
            
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

        # # Save the model at different instances
        # if epoch % 30 == 0:
        #     torch.save(model.module.state_dict(), os.path.join(args.model_dir, args.model_name+str(args.num_modality)+'_epcoh_'+str(epoch)+'.pth'))
        #     print(f"Model saved at epoch {epoch+1} with train loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for batch_idx, (inputs, targets) in enumerate(test_dataloader):

                # Move all inputs and targets to the device
                inputs = [x.to(device) for x in inputs]
                targets = [x.to(device) for x in targets]

                # Forward pass
                reconstructions, masks = model(inputs)

                # Compute loss for each modality
                losses = []
                for rec, target, mask in zip(reconstructions, targets, masks):
                    # Reshape mask to match input dimensions
                    mask_reshaped = mask.view(mask.shape[0], 1, args.image_size[0] // args.patch_size, args.image_size[1] // args.patch_size)
                    mask_reshaped = mask_reshaped.repeat(1, target.size(1), args.patch_size, args.patch_size)
                    
                    # Compute loss only on masked regions
                    loss = criterion(rec * (1 - mask_reshaped), target * (1 - mask_reshaped))
                    losses.append(loss)
                
                # Combine losses
                loss = sum(losses)
                test_loss += loss.item()
                
                # Log input data and reconstructions to TensorBoard
                grid_data = []
                for input_data, rec, target in zip(inputs, reconstructions, targets):
                    grid_data.extend([input_data, rec, target])
                
                grid = make_grid(torch.cat(grid_data, dim=0), nrow=len(inputs))
                writer.add_image(f"Epoch {epoch}, Batch {batch_idx}", grid, epoch * len(test_dataloader) + batch_idx)

            avg_test_loss = test_loss / len(test_dataloader)
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Test Loss: {avg_test_loss:.4f}")
            writer.add_scalar('Test Loss', avg_test_loss, epoch)

            # Save the model if the current test loss is lower than the best one seen so far
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.module.state_dict(), os.path.join(args.model_dir, args.model_name+str(args.num_modality)+'.pth'))
                print(f"New best model saved at epoch {epoch+1} with test loss: {best_test_loss:.4f}")


    # Save the final model with a unique name
    final_model_path = os.path.join(args.model_dir, f'multimae_dynamic_final_{args.num_modality}_{args.num_epochs}.pth')
    torch.save(model.module.state_dict(), final_model_path)
    print(f"Final model saved at: {final_model_path}")

    print("Training completed!")

if __name__ == "__main__":
    main()   

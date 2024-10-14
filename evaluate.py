import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import argparse
from DenoMAE import DenoMAE
from datagen import DenoMAEDataGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="DenoMAE Evaluation Script")
    parser.add_argument("--test_path", type=str, 
                    default='/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/unlabeled/test/',
                    help="Path to test data")
    parser.add_argument("--model_path", type=str, 
                default='/mnt/d/OneDrive - Rowan University/RA//Fall 24/DenoMAE/models/multimae_model_dynamic.pth',
                help="Path to saved model")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save output images")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for evaluation")
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224), help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--in_chans", type=int, default=3, help="Number of input channels")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=12, help="Depth of the encoder")
    parser.add_argument("--decoder_depth", type=int, default=4, help="Depth of the decoder")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_modality", type=int, default=5, help="Number of modalities")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for data loader")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "output"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "ground_truth"), exist_ok=True)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = DenoMAE(args.num_modality, args.image_size[0], args.patch_size, args.in_chans, args.embed_dim, args.encoder_depth, args.decoder_depth, args.num_heads)
    
    # Load the saved model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Set up the data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = DenoMAEDataGenerator(
        noisy_image_path=os.path.join(args.test_path, 'noisyImg/'),
        noiseless_img_path=os.path.join(args.test_path, 'noiseLessImg/'),
        noisy_signal_path=os.path.join(args.test_path, 'noisySignal/'),
        noiseless_signal_path=os.path.join(args.test_path, 'noiselessSignal/'),
        noise_path=os.path.join(args.test_path, 'noise/'),
        image_size=args.image_size,
        transform=transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)

    # Evaluation loop
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            # Move inputs and targets to the device
            inputs = [x.to(device) for x in inputs]
            targets = [x.to(device) for x in targets]

            # Forward pass
            reconstructions, masks = model(inputs)
            print(masks[0].shape)

            # # Save images
            # for i in range(len(inputs[0])):
            #     # Save input images
            #     save_image(inputs[0][i], os.path.join(args.output_dir, "input", f"batch_{batch_idx}_sample_{i}.png"))

            #     # Save reconstructed (output) images
            #     save_image(reconstructions[0][i], os.path.join(args.output_dir, "output", f"batch_{batch_idx}_sample_{i}.png"))

            #     # Save ground truth images
            #     save_image(targets[0][i], os.path.join(args.output_dir, "ground_truth", f"batch_{batch_idx}_sample_{i}.png"))

            # print(f"Processed batch {batch_idx+1}/{len(test_dataloader)}")

    print("Evaluation completed. Results saved in:", args.output_dir)

if __name__ == "__main__":
    main()
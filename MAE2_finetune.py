import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from functools import partial
from tqdm import tqdm
import os
from MAE_2_0 import MaskedAutoencoderViT
from torch.utils.data import DataLoader, Dataset

    
class MAEDownstreamClassifier(nn.Module):
    def __init__(self, mae_model, num_classes, freeze_encoder=True):
        """
        Downstream classifier using a pre-trained MAE model as the backbone
        
        Args:
            mae_model (MaskedAutoencoderViT): Pre-trained MAE model
            num_classes (int): Number of classes for classification
            freeze_encoder (bool): Whether to freeze the MAE encoder weights
        """
        super().__init__()
        
        # Use the encoder from the MAE model
        self.encoder = mae_model
        
        # Freeze the encoder weights if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(mae_model.pos_embed.shape[-1], 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features using the MAE encoder
        # We only want the [CLS] token representation
        with torch.no_grad():
            features, _, _ = self.encoder.forward_encoder(x, mask_ratio=0)
        
        # Take the [CLS] token representation (first token)
        cls_token = features[:, 0, :]
        
        # Pass through classification head
        logits = self.classification_head(cls_token)
        
        return logits
    
def train_classifier(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training Classifier")
    for batch in progress_bar:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({'loss': loss.item(), 'accuracy': 100. * correct / total})

    return total_loss / len(train_loader), 100. * correct / total

def evaluate_classifier(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating Classifier")
        for batch in progress_bar:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({'loss': loss.item(), 'accuracy': 100. * correct / total})

    return total_loss / len(test_loader), 100. * correct / total

def main():
    # Configuration
    config = {
        'train_data_path': '/mnt/d/OneDrive - Rowan University/RA/Fall 24/DenoMAE_2.0/sample_generation/data/labeled/train/noiseLessImg/',
        'test_data_path': '/mnt/d/OneDrive - Rowan University/RA/Fall 24/DenoMAE_2.0/sample_generation/data/labeled/test/noiseLessImg/',
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 1e-3,
        'num_classes': 10,  # Update for your classification task
        'pretrained_model_path': 'models/mae_best_model.pth'
    }

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data augmentation and preprocessing
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=config['train_data_path'], transform=transform)
    test_dataset = datasets.ImageFolder(root=config['test_data_path'], transform=transform)

    train_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    # Load pretrained MAE encoder
    base_encoder = MaskedAutoencoderViT(
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

    checkpoint = torch.load(config['pretrained_model_path'], map_location=device)
    base_encoder.load_state_dict(checkpoint['model_state_dict'])

    # Initialize the downstream classifier
    model = MAEDownstreamClassifier(base_encoder, config['num_classes']).to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_acc = train_classifier(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate_classifier(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # Save the fine-tuned classifier
    torch.save(model.state_dict(), 'models/downstream_classifier.pth')

if __name__ == "__main__":
    main()

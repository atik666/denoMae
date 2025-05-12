import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from DenoMAE import DenoMAE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

class FineTunedDenoMAE(nn.Module):
    def __init__(self, pretrained_model, num_classes, freeze_encoder=True):
        super().__init__()
        self.backbone = pretrained_model

        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.patch_embeds[0](x)
        x = x + self.backbone.pos_embed[:, 1:, :]
        x = self.backbone.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x

def get_args():
    parser = argparse.ArgumentParser(description='Fine-tune DenoMAE')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--in_chans', type=int, default=3, help='Number of input channels')
    parser.add_argument('--mask_ratio', type=float, default=0.0, help='Mask ratio for DenoMAE')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--encoder_depth', type=int, default=12, help='Depth of encoder')
    parser.add_argument('--decoder_depth', type=int, default=4, help='Depth of decoder')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--num_modality', type=int, default=1, help='Number of modalities')
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='Path to pre-trained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save_model_path', type=str, default='models/best_finetuned_model.pth', help='Path to save the best model')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use (e.g., cuda or cpu)')
    parser.add_argument('--confusion_matrix_path', type=str, default='results/confusion_matrix_5.png', 
                        help='Path to save the confusion matrix')
    return parser.parse_args()

def main():
    args = get_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Create directory for results if it doesn't exist
    os.makedirs(os.path.dirname(args.confusion_matrix_path), exist_ok=True)

    # Load pre-trained DenoMAE model
    pretrained_model = DenoMAE(args.num_modality, args.img_size, args.patch_size, args.in_chans, args.mask_ratio,
                               args.embed_dim, args.encoder_depth, args.decoder_depth, args.num_heads)
                               
    pretrained_model.load_state_dict(torch.load(args.pretrained_model_path))
    print(f"Pre-trained model loaded successfully from {args.pretrained_model_path}")

    # Create fine-tuned model
    model = FineTunedDenoMAE(pretrained_model, args.num_classes)
    model = model.to(device)

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=f'{args.data_path}/train/', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'{args.data_path}/test/', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Evaluation function
    def evaluate(model, data_loader, get_confusion=False):
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if get_confusion:
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        if get_confusion:
            return 100. * correct / total, all_preds, all_labels
        return 100. * correct / total
    
    def plot_confusion_matrix(y_true, y_pred, class_names, filepath, accuracy):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Test Accuracy: {accuracy:.2f}%')
        plt.colorbar()
        
        # Add class names to the axes
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations in the cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save the figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved at {filepath}")

    # Training loop
    best_acc = 0
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 100 == 99:
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.3f}, Acc: {100. * correct / total:.3f}%')
                running_loss = 0.0

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_acc, test_preds, test_labels = evaluate(model, test_loader, get_confusion=True)
            print(f'Epoch {epoch + 1}: Test Acc: {test_acc:.3f}%')

            # Save the best model and its confusion matrix
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), args.save_model_path)
                print(f'New best model saved with accuracy: {best_acc:.3f}%')
                
                # Get class names from the dataset
                class_names = test_dataset.classes
                
                # Plot and save the confusion matrix
                plot_confusion_matrix(test_labels, test_preds, class_names, 
                                     args.confusion_matrix_path, test_acc)

    print(f'Training completed. Best accuracy: {best_acc:.3f}%')

    # Final Evaluation
    model.load_state_dict(torch.load(args.save_model_path))
    final_acc, final_preds, final_labels = evaluate(model, test_loader, get_confusion=True)
    print(f'Final Test Accuracy: {final_acc:.3f}%')
    
    # Get class names from the dataset
    class_names = test_dataset.classes
    
    # Generate final confusion matrix
    plot_confusion_matrix(final_labels, final_preds, class_names, 
                         args.confusion_matrix_path, final_acc)

if __name__ == '__main__':
    main()

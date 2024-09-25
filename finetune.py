import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from MultiMAE import MultiMAE
import torchvision.models as models

class FineTunedMultiMAE(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.backbone = pretrained_model
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Use only one modality for classification
        x = self.backbone.patch_embed1(x)
        x = x + self.backbone.pos_embed[:, 1:, :]
        x = self.backbone.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x

# Hyperparameters
img_size = 224
patch_size = 16
in_chans = 3
embed_dim = 768
encoder_depth = 12
decoder_depth = 4
num_heads = 12
batch_size = 32
num_epochs = 100
learning_rate = 1e-4
num_classes = 10  # Adjust based on your dataset

# Load pre-trained MultiMAE model
pretrained_model = MultiMAE(img_size, patch_size, in_chans, embed_dim, encoder_depth, decoder_depth, num_heads)
pretrained_model.load_state_dict(torch.load('models/multimae_model.pth'))

# Create fine-tuned model
model = FineTunedMultiMAE(pretrained_model, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Dataset and DataLoader (example using CIFAR-10)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

path = '/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/labeled/0.5dB/'
train_dataset = datasets.ImageFolder(root=path+'train/', transform=transform)
test_dataset = datasets.ImageFolder(root=path+'test/', transform=transform)

train_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# Training loop
best_acc = 0
for epoch in range(num_epochs):
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

    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        test_acc = evaluate(model, test_loader)
        print(f'Epoch {epoch + 1}: Test Acc: {test_acc:.3f}%')
        
        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'models/best_finetuned_multimae_model.pth')
            print(f'New best model saved with accuracy: {best_acc:.3f}%')

print(f'Training completed. Best accuracy: {best_acc:.3f}%')

# Final Evaluation
model.load_state_dict(torch.load('models/best_finetuned_multimae_model.pth'))
final_acc = evaluate(model, test_loader)
print(f'Final Test Accuracy: {final_acc:.3f}%')
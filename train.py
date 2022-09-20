import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import ImageFile
import torchvision
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets  
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Type123Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 3
num_classes = 3
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Simple Identity class that let's input pass without changes
# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()

#     def forward(self, x):
#         return x

# Load pretrain model & modify it
pretrained_weight = ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=pretrained_weight)

"""do finetuning then set requires_grad = False
Remove these two lines if want to train entire model,
and only want to load the pretrain weights.
for param in model.parameters():
     param.requires_grad = False"""

# model.avgpool = Identity()
# model.classifier = nn.Sequential(
#     nn.Linear(512, 100), 
#     nn.ReLU(),
#     nn.Linear(100, num_classes))
model.fc = nn.Linear(in_features=2048, out_features=3, bias=False)
model.to(device)
# print(model)


# Load Data
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((1.4609, 1.9683, 1.6960), (0.5502, 0.2062, 0.3943))])
train_ds = Type123Dataset(
    csv_file='drive/MyDrive/for_train/fortypes.csv', root_dir='drive/MyDrive/for_train/training/', transform=tfm)

# train_ds, val_ds = random_split(dataset, [1000, 480])
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
# val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)
# Loss and optimizer
criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):

    model.train()
    losses = []
    
    for batch_idx, (data, targets) in enumerate(train_dl):
        # send data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        # calculate loss
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        # set gradient back to zero
        optimizer.zero_grad()

    print(f"Epoch [{epoch}], Loss {(sum(losses)/len(losses)):.4f}")

# Check accuracy
def validate_accuracy(dl, model):

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
      # send data to cuda if possible
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, pred = scores.max(1)
            num_correct += (pred == y).sum() 
            num_samples += pred.size(0)

        print(f"{(float(num_correct)/float(num_samples)):.4f}")
    
    model.train()
validate_accuracy(train_dl, model)

# validate_accuracy(val_dl, model)

torch.save(model.state_dict(), 'intel_screening_with_pretrained_resnet50.pth')
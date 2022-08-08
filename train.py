import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as tt
from customdataset import Type123Dataset


# 1-1339.jpg removed
class Block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet architecture are in these lines
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        """Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        we need to adapt the Identity (skip connection) so it will be able to be added
        to the layer that's ahead"""
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(intermediate_channels * 4))

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is 4
        self.in_channels = intermediate_channels * 4

        
        """ For first resnet layer: 256 will be mapped to 64 as intermediate layer,
         then finally back to 256. Hence no identity downsample is needed, since stride = 1,
         and also same amount of channels."""
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

# hyperparameters
in_channels = 3
num_classes = 3
learning_rate = 0.01
batch_size = 32
num_epochs = 5

# load data
dataset = Type123Dataset(
    csv_file='fortypes.csv', root_dir='train', transform=tt.ToTensor())

train_ds, val_ds = random_split(dataset, [1000, 480])
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(dataset=val_ds, batch_size=batch_size*2, shuffle=False)


def ResNet50(img_channel=3, num_classes=3):
    return ResNet(Block, [3, 4, 6, 3], img_channel, num_classes)

# model and to gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50().to(device)

# loss and optimizer
criterion = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    for batch_idx, (data, labels) in enumerate(train_dl):
        # send data to gpu
        data = data.to(device=device)
        labels = labels.to(device=device)
        # forward pass
        scores = model(data)
        # calculate loss
        loss = criterion(scores, labels)
        # backward pass
        loss.backward()
        # update weight
        optimizer.step()
        # set gradient back to zero
        optimizer.zero_grad()

# check accuracy
def check_accuracy(dl, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return float(num_correct)/float(num_samples)

val_acc = check_accuracy(val_dl, model)
print(f'Validation data accuracy: {val_acc:.4f}')

torch.save(model.state_dict, 'Intel_MobileODT_screening_with_ResNet50.pth')

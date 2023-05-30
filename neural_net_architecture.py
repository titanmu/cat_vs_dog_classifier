from torch import nn
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# class AdvancedNeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#
#         # Pooling layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # Fully connected layers
#         self.fc1 = nn.Linear(6272, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 2)
#
#         # Dropout layer
#         self.dropout = nn.Dropout(p=0.5)
#
#     def forward(self, x):
#         # Convolutional layers with batch normalization and ReLU activation
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.pool(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.pool(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = self.pool(x)
#
#         # Flatten and pass through fully connected layers with dropout
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         x = self.fc3(x)
#
#         return x

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(256)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layer
        self.fc1 = nn.Linear(256*59*59, 84)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.bn4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.bn5(x))
        x = F.relu(self.pool(x))

        # x = x.view(-1, 256*59*59)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x





import torch
import torch.nn as nn
from torchvision import models

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1) for i in range(num_layers)])

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        densenet = models.densenet121(pretrained=True)
        
        # Extract layers from the pre-trained model
        self.conv1 = densenet.features.conv0
        self.norm1 = densenet.features.norm0
        self.relu = densenet.features.relu0
        self.pool = densenet.features.pool0

        # Define dense blocks and transition layers
        self.dense_block1 = DenseBlock(64, 32, 6)
        self.transition1 = TransitionLayer(64 + 6 * 32, 128)
        self.dense_block2 = DenseBlock(128, 32, 12)
        self.transition2 = TransitionLayer(128 + 12 * 32, 256)
        self.dense_block3 = DenseBlock(256, 32, 24)
        self.transition3 = TransitionLayer(256 + 24 * 32, 512)
        self.dense_block4 = DenseBlock(512, 32, 16)

        # Get the actual output size of DenseNet features
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)
            features_output = self.dense_block4(self.transition3(self.dense_block3(self.transition2(self.dense_block2(self.transition1(self.dense_block1(self.pool(self.relu(self.norm1(self.conv1(sample_input)))))))))))
            self.input_size = features_output.view(features_output.size(0), -1).size(1)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d((7, 7))  # Adjust the size accordingly

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, 512),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 6)
        )
        

    def forward(self, x):
        # Initial layers
        x = self.pool(self.relu(self.norm1(self.conv1(x))))

        # Dense blocks and transition layers
        x = self.dense_block1(x)
        x = self.transition1(x)
        x = self.dense_block2(x)
        x = self.transition2(x)
        x = self.dense_block3(x)
        x = self.transition3(x)
        x = self.dense_block4(x)

        # Global average pooling
        x = self.global_avg_pooling(x)
        
        # Flatten the features
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.classifier(x)
        
        return x

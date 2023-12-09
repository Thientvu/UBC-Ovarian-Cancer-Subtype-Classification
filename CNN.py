import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=10,
                               kernel_size=(3,3),
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=10,
                               out_channels=20,
                               kernel_size=(5,5),
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=20,
                               out_channels=40,
                               kernel_size=(7,7),
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=40,
                               out_channels=60,
                               kernel_size=(9,9),
                               padding=1)
        self.conv5 =  nn.Conv2d(in_channels=60,
                               out_channels=80,
                               kernel_size=(9,9),
                               padding=1)
        self.conv6 =  nn.Conv2d(in_channels=80,
                               out_channels=100,
                               kernel_size=(11,11),
                               padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4900,500)
        self.linear2 = nn.Linear(500,500)
        self.linear3 = nn.Linear(500,6)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        

    def forward(self, X):
        out = self.conv1(X)
        out = self.pool(out)
        out = self.ReLU(out)

        out = self.conv2(out)
        out = self.pool(out)
        out = self.ReLU(out)

        out = self.conv3(out)
        out = self.pool(out)
        out = self.ReLU(out)

        out = self.conv4(out)
        out = self.pool(out)
        out = self.ReLU(out)

        out = self.conv5(out)
        out = self.pool(out)
        out = self.ReLU(out)

        out = self.conv6(out)
        out = self.pool(out)
        out = self.ReLU(out)

        out = self.flatten(out)
        
        out = self.linear1(out)
        out = self.ReLU(out)
        out = self.linear2(out)
        out = self.ReLU(out)
        out = self.linear3(out)
        
        out = self.softmax(out)
        return out
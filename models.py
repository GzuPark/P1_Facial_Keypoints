## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        I.xavier_uniform_(self.conv1.weight)
        self.conv1.bias.data.fill_(0.0)
        # (224-5)/1+1 = 220
        # (32, 220, 220)
        
        # After maxpooling
        # (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        I.xavier_uniform_(self.conv2.weight)
        self.conv2.bias.data.fill_(0.0)
        # (110-5)/1+1 = 106
        # (64, 106, 106)
        
        # After maxpooling
        # (64, 53, 53)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        I.xavier_uniform_(self.conv3.weight)
        self.conv3.bias.data.fill_(0.0)
        # (53-5)/1+1 = 49
        # (128, 49, 49)
        
        # After maxpooling
        # (128, 24, 24)
        self.conv4 = nn.Conv2d(128, 256, 5, 1)
        I.xavier_uniform_(self.conv4.weight)
        self.conv4.bias.data.fill_(0.0)
        # (24-5)/1+1 = 20
        # (256, 20, 20)
        
        # After maxpooling
        # (256, 10, 10)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.5)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.5)
        
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.5)
        
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.5)
        
        # from (256, 10, 10)
        self.fc1 = nn.Linear(256*10*10, 1000)
        self.dropout5 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(1000, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout5(x)
        
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

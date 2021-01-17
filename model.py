#importing dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transform
class CNN(nn.Module):
    def __init__(self,in_channels=3,num_classes=2):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=8,
                               kernel_size=(5,5),
                               stride =(1,1),
                               padding=(1,1)              # nout = [(nin + 2p - k)/s] + 1
                              )
        self.batnm1 = nn.BatchNorm2d(8)                   # [(224 + 2 . 1 - 5) / 1] + 1 = 222
        self.pool = nn.MaxPool2d(kernel_size=(2,2),       # after maxpooling 222 / 2 = 111
                                 stride=(2,2)
                                )                         #[(111 + 2 . 1 - 3) / 2] + 1 = 56
        self.conv2 = nn.Conv2d(in_channels=8,
                              out_channels=32,            # after maxpooling 56/2 = 28
                              kernel_size=(3,3),
                               stride=(2,2),
                               padding=(1,1)
                              )
        self.batnm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32,
                              out_channels=64,            # (28 + 2 . 1 - 3) / 1] + 1 = 28
                              kernel_size=(3,3),
                               stride=(1,1),              # after maxapooling 28/2 = 14
                               padding=(1,1)
                              )
        self.batnm3 = nn.BatchNorm2d(64) #on forward pass we will again add another Maxpooling layer here
        self.fc1 = nn.Linear(64*14*14 ,num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.batnm1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.batnm2(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.batnm3(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

model = CNN()
x = torch.randn(128,3,224,224)
print(x.shape)
print(model(x).shape)



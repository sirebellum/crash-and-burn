# Description: Contains the CNN model used to locate and rotate a qr code
from torch import nn
import torch.nn.functional as F

# CNN to locate and rotate a qr code
class Net(nn.Module):
    
        def __init__(self):
            super(Net, self).__init__()
    
            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv5 = nn.Conv2d(128, 256, 3, padding=1)

            # FC
            self.fc = nn.Linear(256*16*16, 1024)

            # BBox
            self.fc_bbox = nn.Linear(1024, 4)

            # Rot
            self.fc_rot = nn.Linear(1024, 2)

        def forward(self, x):
            # Convolutional layers
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv4(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv5(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 256*16*16)

            # FC
            x = F.relu(self.fc(x))

            # Bbox
            bbox = self.fc_bbox(x)
            bbox = F.sigmoid(bbox)

            # Rot
            rot = self.fc_rot(x)
            rot = F.softmax(rot, dim=1)

            return bbox, rot

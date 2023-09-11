# Description: Contains the CNN model used to locate and rotate a qr code
import torch
from torch import nn
import torch.nn.functional as F

IMAGE_SIZE=256

# CNN to locate and rotate a qr code
class Net(nn.Module):
    
        def __init__(self):
            super(Net, self).__init__()
    
            # Convolutional layers
            self.n_layers = 5
            self.convs = nn.ModuleList()
            for n in range(self.n_layers):
                if n==0:
                    self.convs.append(nn.Conv2d(3, 16, 3, padding=1))
                else:
                    self.convs.append(nn.Conv2d(16*(2**(n-1)), 16*(2**n), 3, padding=1))

            # FC
            self.flattened_size = 16*(2**(self.n_layers-1))*((IMAGE_SIZE//(2**self.n_layers))**2)
            self.fc = nn.Linear(self.flattened_size, 128)

            # BBox
            self.fc_bbox = nn.Linear(128, 4)

            # Rot
            self.fc_rot = nn.Linear(128, 2)

        def forward(self, x):
            # Convolutional layers
            for conv in self.convs:
                x = F.relu(conv(x))
                x = F.max_pool2d(x, 2)
            
            # Flatten
            x = x.view(-1, self.flattened_size)

            # FC
            x = F.relu(self.fc(x))

            # Bbox
            bbox = self.fc_bbox(x)
            bbox = F.sigmoid(bbox)

            # Rot
            rot = self.fc_rot(x)
            rot = F.tanh(rot)

            # Concatenate the outputs
            out = torch.cat((bbox, rot), 1)

            return out

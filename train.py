# Description: Train a torch CNN to locate and rotate a qr code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from dataset import LandingDataset

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

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

            # Fully connected layers
            self.fc1 = nn.Linear(256*16*16, 4)

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

            # Fully connected layers
            x = x.view(-1, 256*16*16)
            x = self.fc1(x)
            x = F.sigmoid(x)

            return x
        
# Custom loss function
def custom_loss(outputs, labels):

    # Get the labels
    bbox_label = labels["bbox"].to(device).float()

    # Get the preds
    bbox_pred = outputs

    # Calculate the MSE loss
    loss = F.mse_loss(bbox_pred, bbox_label)

    return loss

# Train the model
def train():
         
        # Create the dataset
        dataset = LandingDataset(n_samples=10000, image_size=(3,512,512))
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
        # Create the model
        model = Net().to(device)
    
        # Create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
        # Create the tensorboard writer
        writer = SummaryWriter()

        # Train the model
        for _ in range(10):
            bar = tqdm(dataloader)
            for data in bar:
    
                # Get the inputs
                inputs, labels = data
    
                # Zero the parameter gradients
                optimizer.zero_grad()
    
                # Forward + backward + optimize
                outputs = model(inputs.to(device))
                loss = custom_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                # Update the progress bar
                bar.set_description("Loss: %.4f" % loss.item())

                # Log the loss
                writer.add_scalar("Loss", loss.item())
    
        # Save the model as torchscript jit
        model.eval().to("cpu")
        example = torch.rand(1, 3, 512, 512)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save("model.pt")

if __name__ == "__main__":
    train()

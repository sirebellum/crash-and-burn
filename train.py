# Description: Train a torch CNN to locate and rotate a qr code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from dataset import LandingDataset
from models import Net

# Create the dataset
dataset = LandingDataset(n_samples=10000, image_size=(3,512,512))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
        
# Custom loss function
def custom_loss(outputs, labels):

    # Get the labels
    bbox_label = labels["bbox"].to(device).float()
    rot_label = labels["rot"].to(device).float()

    # Get the preds
    bbox_pred = outputs[0]
    rot_pred = outputs[1]

    # Calculate the bbox loss
    loss = F.mse_loss(bbox_pred, bbox_label)

    # Calculate the rot loss
    loss += F.mse_loss(rot_pred, rot_label)

    return loss

# Train the model on bounding boxes
def train():
         
    # Create the model
    model = Net().to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Create the tensorboard writer
    writer = SummaryWriter()

    # Train the model
    for _ in range(24):
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

    # Save the model
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    train()

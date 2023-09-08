# Description: Test the rotation and location prediction of the model
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
import cv2
import numpy as np

from dataset import LandingDataset

# Load dataset
dataset = LandingDataset(n_samples=10, image_size=(3,512,512))

# Load the model
model = torch.jit.load('model.pt')

for image, labels in dataset:

    # Get the labeks
    rot_label = labels['rot'].numpy() * 360
    bbox_label = labels['bbox'].numpy() * 512
    bbox_label = bbox_label.astype('int')
    rot_label = rot_label.astype('int')

    # Get the preds
    with torch.no_grad():
        pred = model(image.unsqueeze(0))
    bbox_pred = (pred[0].squeeze().numpy()*512).astype('int')
    rot_pred = (pred[1].squeeze().numpy()).astype('int')

    # Convert sin and cos rot pred to degrees
    rot_pred = int(np.arctan2(rot_pred[0], rot_pred[1]) * 180 / np.pi)
    rot_label = int(np.arctan2(rot_label[0], rot_label[1]) * 180 / np.pi)

    # Create bounding boxes on the image
    image = image.numpy().transpose(1,2,0)*255
    image = image.astype('uint8').copy()
    image = cv2.rectangle(
        image,
        (bbox_label[0], bbox_label[1]),
        (bbox_label[2], bbox_label[3]),
        (0,255,0),
        2
    )
    image = cv2.rectangle(
        image,
        (bbox_pred[0], bbox_pred[1]),
        (bbox_pred[2], bbox_pred[3]),
        (255,0,0),
        2
    )

    # Rotate the image
    image = rotate(torch.tensor(image.transpose(2,0,1)), -rot_pred)
    image = image.numpy().transpose(1,2,0)

    # Display the image
    cv2.imshow('image', image)
    cv2.waitKey(0)

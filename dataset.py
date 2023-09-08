# Description: Create a generator dataset for use with torch model
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
import cv2
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm

# Class to generate images with a landing pad on a background
class LandingDataset(Dataset):

    def __init__(self, n_samples=10000, image_size=(3,1024,1024)):
        self.n_samples = n_samples
        self.image_size = image_size

        # Load the background images
        background_paths = glob('backgrounds/*.jpg')
        self.backgrounds = []
        for path in background_paths:
            img = cv2.imread(path).transpose(2,0,1)
            img = torch.tensor(img).float()/255
            self.backgrounds.append(img)
        
        # Load the objects
        object_paths = glob('objects/*.jpg')
        object_paths += glob('objects/*.png')
        self.objects = []
        for path in object_paths:
            # Load the image
            img = cv2.imread(path).transpose(2,0,1)
            img = torch.tensor(img).float()/255
            self.objects.append(img)

        # Buffer of landing pad images
        self.landing_buffer = []

        # Generate the landing pad images
        for i in tqdm(range(n_samples)):
            self.landing_buffer.append(self.gen_full(size=image_size))

    # Generate a full image with a landing pad
    def gen_full(self, size=(3,1024,1024)):

        # Generate the landing pad of random size
        landing, labels = self.gen_landing(size=size)

        # Generate the background
        background = self.gen_background(size=size)

        # Generate the object map
        objects = self.gen_objects(size=size)

        # Add the object to the background
        background = background + objects

        # Add the landing pad to the background
        background = background + landing

        # Clip the background
        background = torch.clamp(background, 0, 1)

        return background.float(), labels

    # Generate a random object map
    def gen_objects(self, size=(3,1024,1024)):

        # How many objects?
        n_objects = torch.randint(0, 10, (1,)).item()

        # What size?
        max_size = min(size[1], size[2])//8
        min_size = max_size//4
        w = torch.randint(min_size, max_size, (n_objects,))
        h = torch.randint(min_size, max_size, (n_objects,))

        # Where on the image?
        x = torch.randint(0, size[1]-max_size, (n_objects,))
        y = torch.randint(0, size[2]-max_size, (n_objects,))

        # What objects?
        object_idxs = torch.randint(0, len(self.objects), (n_objects,))

        # Generate the object map
        objects = torch.zeros(size)
        for i in range(n_objects):
            object = self.objects[object_idxs[i]]
            # Resize the object
            object = F.interpolate(object.unsqueeze(0), size=(w[i],h[i]), mode="bilinear")
            object[object == 1] = 0
            objects[:, x[i]:x[i]+w[i], y[i]:y[i]+h[i]] = object.squeeze(0)

        return objects

    # Generate a random background image
    def gen_background(self, size=(3,1024,1024)):

        # Select a random background
        background = self.backgrounds[torch.randint(0, len(self.backgrounds), (1,)).item()]

        # Resize the background
        background = F.interpolate(background.unsqueeze(0), size=size[1:])

        return background.squeeze(0)

    # Generate a landing pad with 3 white squares in a triangle
    # Maintain a small border around the edge of the pad
    # Overlay on larger background
    def gen_landing(self, size=(3,1024,1024)):

        landing_size = torch.randint(size[1]//16, size[1]//4, (1,)).item()
        
        # Landing pad assets
        background = torch.zeros(size)
        landing_pad_shape = (landing_size, landing_size)
        landing_pad = torch.zeros(landing_pad_shape) - 1
        edge_buffer = landing_size//8

        # Generate the landing pad
        landing_pad[edge_buffer:edge_buffer+landing_size//3, edge_buffer:edge_buffer+landing_size//3] = 1
        landing_pad[edge_buffer:edge_buffer+landing_size//3, -edge_buffer-landing_size//3:-edge_buffer] = 1
        landing_pad[-edge_buffer-landing_size//3:-edge_buffer, edge_buffer:edge_buffer+landing_size//3] = 1

        # Randomly rotate the landing pad
        rotation = torch.randint(0, 360, (1,))
        landing_pad = rotate(landing_pad.unsqueeze(0), rotation.item())

        # Place randomly on the background
        x = torch.randint(0, size[1]-landing_size, (1,))
        y = torch.randint(0, size[2]-landing_size, (1,))
        background[:,
                   x.item():x.item()+landing_size,
                   y.item():y.item()+landing_size] = landing_pad
        
        # Get bounding box for landing pad
        x_min = x.item()
        x_max = x.item()+landing_size
        y_min = y.item()
        y_max = y.item()+landing_size

        bounding_box = torch.tensor([y_min, x_min, y_max, x_max]) / size[1]

        return background, {"rot": rotation.squeeze()/360,
                            "bbox": bounding_box,}

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.landing_buffer[idx]

if __name__ == "__main__":
    # Generate a dataset
    dataset = LandingDataset(n_samples=1000)

    # Display some samples
    import matplotlib.pyplot as plt
    for i in range(10):
        plt.imshow(dataset[i][0].numpy().transpose(1,2,0))
        plt.show()

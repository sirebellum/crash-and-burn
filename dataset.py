import cv2
import numpy as np
import glob
from random import sample
from math import pi
import math
from tqdm import tqdm

class QRDataset:

    def __init__(self, n_samples=10000, image_size=(64*3,64*3,3)):
        self.image_size = image_size

        backgrounds = glob.glob("backgrounds/*")
        bak_size = image_size
        self.bak_images = np.empty((len(backgrounds),*bak_size), dtype="int16")
        for n, b in enumerate(backgrounds):
            self.bak_images[n] = cv2.resize(cv2.imread(b), bak_size[:2])

        # Generate objects
        objects = glob.glob("objects/*")
        obj_images = []
        print("Generating objects...")
        for o in objects:
            for pert in range(1000):
                obj = cv2.imread(o)
                obj = cv2.resize(obj, (np.random.randint(8,64), np.random.randint(8,64)))
                obj = self.rotate_image(obj, np.random.randint(360))
                obj_images.append(obj)


        # Generate reference squares
        print("Generating squares...")
        sizes = (8*2,8*4,8*8,8*12)
        even_rots = (0,90,180,240,360)
        self.code_images = []
        for pert in range(10000):
            codesize = sample(sizes, 1)[0]
            rotation = np.random.randint(360)
            code = self.genCode(size=codesize, rotation=rotation)
            self.code_images.append([code, codesize, rotation])

        # Overlay qrcode on random backgrounds with random orientation
        self.images = np.empty((n_samples, *image_size), dtype="int16")
        self.rotations = []
        self.sections = []
        self.n_sections = 9
        for n in tqdm(range(n_samples)):

            # Select random background and rotate
            bak = self.bak_images[np.random.randint(len(self.bak_images))]
            bak = self.rotate_image(bak, sample(even_rots,1)[0])

            # Add objects
            num_objects = np.random.randint(10)
            objs = sample(obj_images, num_objects)
            for obj in objs:
                indexx = np.random.randint(bak.shape[0]-obj.shape[0])
                indexy = np.random.randint(bak.shape[1]-obj.shape[1])
                bak[indexx:indexx+obj.shape[0],indexy:indexy+obj.shape[1]] = obj

            # Place code randomly on background
            code, codesize, rotation = sample(self.code_images, 1)[0]
            indexx = np.random.randint(bak.shape[0]-codesize)
            indexy = np.random.randint(bak.shape[1]-codesize)
            block = bak[indexx:indexx+codesize,indexy:indexy+codesize]
            code[code==69] = block[code==69]
            bak[indexx:indexx+codesize,indexy:indexy+codesize] = code

            # Calculate which section code is in [0,8]
            sectionx = (indexx+codesize/2)//(bak.shape[0]//3)
            sectiony = (indexy+codesize/2)//(bak.shape[1]//3)
            section = int(sectionx*3+sectiony)

            # noise and stuff
            darken_ratio = 0.5+np.random.random()*0.5
            bak = bak*darken_ratio
            bak += np.random.randint(-5,5,size=bak.shape).astype("int8")
            grayscale = np.random.randint(2)
            if grayscale:
                bak = np.stack([bak.mean(axis=-1)]*3, axis=-1).astype("int16")

            # Finalize
            self.rotations.append([rotation/360, 1-(rotation/360)])
            self.sections.append(section)
            self.images[n] = bak

        self.images[self.images<0] = 0
        self.images = self.images.astype("float16")/255.0

        self.rotations = np.array(self.rotations)

        # Convert labels to one hot
        self.sections = np.array(self.sections)
        unique_cats = set(self.sections)
        assert len(unique_cats) == self.n_sections
        self.onehot = np.empty((self.sections.shape[0],self.n_sections), dtype="bool")
        for n in unique_cats:
            self.onehot[self.sections==n] = [i==n for i in range(self.n_sections)]

    def genCode(self, size=8, rotation=False):
        assert size%8 == 0
        code = np.ones((size,size), dtype="uint8")*255
        edge = size//8
        width = size//4
        code[edge:edge+width,edge:edge+width] = 0
        code[-edge-width:-edge,edge:edge+width] = 0
        code[-edge-width:-edge,-edge-width:-edge] = 0

        # Rotate
        if rotation:
            frame = np.zeros((size*3,size*3), dtype="uint8")
            frame[:,:] = 69
            frame[size:size+size, size:size+size] = code
            code = self.rotate_image(frame, rotation, crop=True)
            code = cv2.resize(code, (size,size))
        return np.stack([code]*3, axis=-1)

    def rotate_image(self, image, angle, crop=True):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Crop out black edges
        def rotatedRectWithMaxArea(w, h, angle):
            """
            Given a rectangle of size wxh that has been rotated by 'angle' (in
            radians), computes the width and height of the largest possible
            axis-aligned rectangle (maximal area) within the rotated rectangle.
            """
            if w <= 0 or h <= 0:
                return 0,0

            width_is_longer = w >= h
            side_long, side_short = (w,h) if width_is_longer else (h,w)

            # since the solutions for angle, -angle and 180-angle are all the same,
            # if suffices to look at the first quadrant and the absolute values of sin,cos:
            sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
            if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
                # half constrained case: two crop corners touch the longer side,
                #   the other two corners are on the mid-line parallel to the longer line
                x = 0.5*side_short
                wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
            else:
                # fully constrained case: crop touches all 4 sides
                cos_2a = cos_a*cos_a - sin_a*sin_a
                wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
            return int(wr),int(hr)

        if crop:
            crop_size = rotatedRectWithMaxArea(image.shape[0], image.shape[1], angle*pi/180)
            crop_start = [(image.shape[0]-crop_size[0])//2, (image.shape[1]-crop_size[1])//2] 
            result = cv2.resize(result[crop_start[0]:crop_start[0]+crop_size[0],
                                     crop_start[1]:crop_start[1]+crop_size[1]],
                                image.shape[:2])

        return result


if __name__ == "__main__":

    qrdataset = QRDataset(n_samples=1000)
    for image, section in zip(qrdataset.images, qrdataset.sections):
        cv2.imshow(f"Section {section}", image.astype("float32"))
        cv2.waitKey(250)
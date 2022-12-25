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
        bak_size = (420,420)
        self.bak_images = np.empty((len(backgrounds),*bak_size,3), dtype="uint8")
        for n, b in enumerate(backgrounds):
            self.bak_images[n] = cv2.resize(cv2.imread(b), bak_size)

        sizes = (8*8,8*16,8*32)
        even_rots = (0,90,180,240,360)

        # Overlay qrcode on random backgrounds with random orientation
        self.images = np.empty((n_samples, *image_size), dtype="uint8")
        self.rotations = []
        self.sections = []
        self.n_sections = 9
        for n in tqdm(range(n_samples)):
            
            # Select random background and rotate
            ran_bak = self.bak_images[np.random.randint(len(self.bak_images))]
            rot_bak = self.rotate_image(ran_bak, sample(even_rots,1)[0])
            
            # Generate reference square
            codesize = sample(sizes, 1)[0]
            rotation = np.random.randint(360)
            code = self.genCode(size=codesize, rotation=rotation)

            # Place randomly on background
            indexx = np.random.randint(rot_bak.shape[0]-codesize)
            indexy = np.random.randint(rot_bak.shape[1]-codesize)
            for x in range(codesize):
                for y in range(codesize):
                    if code[x,y] != 69:
                        rot_bak[indexx+x,indexy+y,0] = code[x,y]
                        rot_bak[indexx+x,indexy+y,1] = code[x,y]
                        rot_bak[indexx+x,indexy+y,2] = code[x,y]

            # Calculate which section code is in [0,8]
            sectionx = (indexx+codesize/2)//(rot_bak.shape[0]//3)
            sectiony = (indexy+codesize/2)//(rot_bak.shape[1]//3)
            section = int(sectionx*3+sectiony)

            self.rotations.append(rotation/360) # Value between 0 and 1
            self.sections.append(section)
            self.images[n] = cv2.resize(rot_bak, image_size[:2])

        self.images = self.images.astype("float32")/255.0

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
        return code

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
        cv2.imshow(f"Section {section}", image)
        cv2.waitKey(250)
import os

import scipy.io as sio
import cv2
import numpy as np


BasePath = os.path.dirname(os.path.dirname(__file__))
FullImgMean = os.path.join(BasePath, "utils", "places_mean_resize.mat")
HeadImgMean = os.path.join(BasePath, "utils", "imagenet_mean_resize.mat")


def transformData(input_img, mean_img):
    transformed_img = cv2.resize(input_img, mean_img.shape[:2]) # Default is bilinear interpolation
    transformed_img = transformed_img - mean_img
    transformed_img = np.transpose(transformed_img, (2,0,1))
    return transformed_img







class GFAnnotation(object):
    def __init__(self, eyes, target, color=(255,0,255)):
        self.eyes = eyes
        self.target = target
        self.color = color

    ##### For display
    def draw(self, image, color=None):
        if self.target[0] < 0 or self.target[0] > 1 or self.target[1] < 0 or self.target[1] > 1:
            return
        if color is None:
            color = self.color

        height, width, _ = image.shape
        p_eyes = (int(self.eyes[0]*width), int(self.eyes[1]*height))
        p_target = (int(self.target[0]*width), int(self.target[1]*height))
        cv2.line(image, p_eyes, p_target, color, 3)
        radius = (width+height) // 15
        cv2.circle(image, p_eyes, radius, (255,0,0))





class GFImage(object):
    ##### Loading data
    def __init__(self, path):
        self.path = path
        self.annotations = []
        self.image = None
        self.annot_image = None

    def addAnnotation(self, eyes_x, eyes_y, target_x, target_y, **kwargs):
        self.annotations.append(GFAnnotation( (float(eyes_x), float(eyes_y)),
                                              (float(target_x), float(target_y)),
                                              **kwargs))

    def getImage(self):
        if self.image is None:
            self.image = cv2.imread(self.path)
        return self.image


    def getHeadImage(self, eye_index=0, **kwargs): # eye_index: which annotation from the list is chosen
        if not hasattr(self, "head_image"):
            img = self.getImage()
            eyes = self.annotations[eye_index].eyes
            w_x = int(np.floor( 0.3 * img.shape[1] ))
            w_y = int(np.floor( 0.3 * img.shape[0] ))
            if w_x%2 == 0:
                w_x += 1
            if w_y%2 == 0:
                w_y += 1
            # head_image is a rectangle of size w_x*w_y around the position of the eyes
            # -> Pixels outside the image are set to (104,117,123)
            # -> Pixels inside the image are copied
            self.head_image = np.ones((w_y,w_x,3), dtype='uint8')
            self.head_image[:,:,0] = 104*np.ones((w_y,w_x),dtype='uint8')
            self.head_image[:,:,1] = 117*np.ones((w_y,w_x),dtype='uint8')
            self.head_image[:,:,2] = 123*np.ones((w_y,w_x),dtype='uint8')
            center = np.floor([eyes[0]*img.shape[1], eyes[1]*img.shape[0]]).astype(int)
            d_x = int(np.floor((w_x-1)/2))
            d_y = int(np.floor((w_y-1)/2))
            tmp = center[0]-d_x;
            img_l   = max(0, tmp);
            delta_l = max(0, -tmp);
            tmp = center[0]+d_x+1;
            img_r   = min(img.shape[1], tmp);
            delta_r = w_x-(tmp-img_r);
            tmp = center[1]-d_y;
            img_t   = max(0, tmp);
            delta_t = max(0, -tmp);
            tmp = center[1]+d_y+1;
            img_b   = min(img.shape[0], tmp);
            delta_b = w_y-(tmp-img_b);
            self.head_image[delta_t:delta_b,delta_l:delta_r,:] = img[img_t:img_b,img_l:img_r,:]
        return self.head_image


    ##### For display
    def getAnnotatedImage(self):
        if self.annot_image is None:
            self.annot_image = self.getImage().copy()
            height, width, _ = self.annot_image.shape
            for annot in self.annotations:
                annot.draw(self.annot_image)
        return self.annot_image

    ##### Preparing data for the network
    @classmethod
    def loadFullImageMean(cls):
        cls.full_img_mean  = sio.loadmat(FullImgMean)['image_mean']
        cls.full_img_mean  = cv2.resize(cls.full_img_mean , (227,227))

    @classmethod
    def loadHeadImageMean(cls):
        cls.head_img_mean  = sio.loadmat(HeadImgMean)['image_mean']
        cls.head_img_mean  = cv2.resize(cls.head_img_mean , (227,227))

    def getFormattedImage(self):
        if not hasattr(self, "full_img_mean"):
            self.loadFullImageMean()
        if not hasattr(self, "full_img"):
            self.formatted_full_img = transformData(self.getImage(), self.full_img_mean)
        return self.formatted_full_img

    def getFormattedHeadImage(self, **kwargs):
        if not hasattr(self, "head_img_mean"):
            self.loadHeadImageMean()
        if not hasattr(self, "head_img"):
            self.formatted_head_img = transformData(self.getHeadImage(**kwargs), self.head_img_mean)
        return self.formatted_head_img

    def getMask(self, eye_index=0):
        if not hasattr(self, "mask"):
            eyes = self.annotations[eye_index].eyes
            self.mask = np.zeros((169,1,1))
            fx = int(np.floor(eyes[0]*13))
            fy = int(np.floor(eyes[1]*13))
            self.mask[13*fy+fx,0,0] = 1

        return self.mask



class GFReader(object):
    """
    Encapsulate the logic for reader the dataset
    """
    def __init__(self, path):
        self.images = {}
        self.path = os.path.realpath(path)
        self.basepath = os.path.dirname(self.path)
        if self.path[-4:] == ".mat":
            self.read_from_mat()
        elif self.path[-4:] == ".txt":
            self.read_from_txt()
        else:
            raise ValueError("Unknown annotation format")


    def read_from_mat(self):
        raise NotImplementedError("TODO")


    def read_from_txt(self):
        # TODO: try Pandas for once
        with open(self.path) as istream:
            for line in istream:
                values = line.rstrip("\n").split(",")
                image_path = values[0]
                if image_path not in self.images:
                    actual_path = os.path.join(self.basepath, *(image_path.split("/")))
                    self.images[image_path] = GFImage(actual_path)
                self.images[image_path].addAnnotation(*values[6:10])


    def getRandom(self):
        return np.random.choice(list(self.images.values()))

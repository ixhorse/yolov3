import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh


class LoadImages():  # for inference
    def __init__(self, path, img_size=416):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.height = img_size

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'File Not Found ' + img_path

        # Padded resize
        img, _, = letterbox(img0, None, height=self.height, mode='test')

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = normalize(img)

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, img_size=416):
        self.cam = cv2.VideoCapture(0)
        self.height = img_size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cam.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, mode='test')

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = normalize(img)

        return img_path, img, img0

    def __len__(self):
        return 0


class LoadImagesAndLabels(Dataset):  # for training
    def __init__(self, img_files, label_files=None, img_size=608, mode='train'):
        self.img_files = img_files
        self.label_files = label_files if label_files else \
                            [x.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]
        self.nF = len(self.img_files)  # number of image files
        self.height = img_size
        self.mode = mode
        self.shuffled_vector = np.random.permutation(self.nF)

        assert self.nF > 0, 'No images found in %s' % path

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.shuffled_vector[index]
        # read img and label
        img = cv2.imread(self.img_files[index])  # BGR
        assert img is not None, 'File Not Found ' + self.img_files[index]
        h, w = img.shape[:2]

        labels = self._load_label(self.label_files[index])
        if self.mode == 'train':
            # hsv
            img = augment_hsv(img, fraction=0.5)
            # pad and resize
            img, labels = letterbox(img, labels, height=self.height, mode='test')
            # Augment image and labels
            img, labels, M = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
            # random left-right flip
            img, labels = random_flip(img, labels, 0.5)
            # color distort
            # img = random_color_distort(img)
        else:
            # pad and resize
            img, labels = letterbox(img, labels, height=self.height, mode=self.mode)

        # show_image(img, labels)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels = np.clip(labels, 0, self.height - 1)
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy()) / self.height

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = normalize(img)

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = torch.from_numpy(img).float()
        labels = labels.float()
        shape = np.array([h,w], dtype=np.float32)
        return (img, labels, shape, self.img_files[index])
    
     @staticmethod
    def collate_fn(batch):
        img, label, hw, path = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), hw, path

    def _load_label(self, label_path):
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 1] = labels0[:, 1] - labels0[:, 3] / 2
            labels[:, 2] = labels0[:, 2] - labels0[:, 4] / 2
            labels[:, 3] = labels0[:, 1] + labels0[:, 3] / 2
            labels[:, 4] = labels0[:, 2] + labels0[:, 4] / 2
        else:
            labels = np.array([])
        return labels

    def __len__(self):
        return self.nF  # number of batches

def augment_hsv(img, fraction):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = img_hsv[:, :, 1].astype(np.float32)
    V = img_hsv[:, :, 2].astype(np.float32)

    a = (random.random() * 2 - 1) * fraction + 1
    S *= a
    if a > 1:
        np.clip(S, a_min=0, a_max=255, out=S)

    a = (random.random() * 2 - 1) * fraction + 1
    V *= a
    if a > 1:
        np.clip(V, a_min=0, a_max=255, out=V)

    img_hsv[:, :, 1] = S.astype(np.uint8)
    img_hsv[:, :, 2] = V.astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
    return img


def letterbox(img, labels, height=416, mode='train', color=(127.5, 127.5, 127.5)):
    """
    resize a rectangular image to a padded square
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    if mode == 'test':
        dw = (max(shape) - shape[1]) / 2  # width padding
        dh = (max(shape) - shape[0]) / 2  # height padding
        left, right = round(dw - 0.1), round(dw + 0.1)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
    else:
        dw = random.randint(0, max(shape) - shape[1])
        dh = random.randint(0, max(shape) - shape[0])
        left, right = dw, max(shape) - shape[1] - dw
        top, bottom = dh, max(shape) - shape[0] - dh
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    interp = np.random.randint(0, 5)
    img = cv2.resize(img, (height, height), interpolation=interp)  # resized, no border
    if labels is not None and len(labels) > 0:
        labels[:, 1] = ratio * (labels[:, 1] + left)
        labels[:, 2] = ratio * (labels[:, 2] + top)
        labels[:, 3] = ratio * (labels[:, 3] + left)
        labels[:, 4] = ratio * (labels[:, 4] + top)

    return img, labels


def random_affine(img, targets=(), degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # apply angle-based reduction
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        np.clip(xy, 0, height, out=xy)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targets

def random_flip(img, labels, px=0.5):
    """
    random horizontal flip
    """
    height = img.shape[0]
    if random.random() < px:
        img = np.fliplr(img).copy()
        if(len(labels) > 0):
            labels[:, 1] = height - labels[:, 1]
            labels[:, 3] = height - labels[:, 3]
            labels[:, [1, 3]] = labels[:, [3, 1]]
    return img, labels


def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    norm = (x - mean) / std
    """
    img = img / 255.0
    #mean = np.array(mean)
    #std = np.array(std)
    #img = (img - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]
    return img.astype(np.float32)


def convert_tif2bmp(p='../xview/val_images_bmp'):
    import glob
    import cv2
    files = sorted(glob.glob('%s/*.tif' % p))
    for i, f in enumerate(files):
        print('%g/%g' % (i + 1, len(files)))
        cv2.imwrite(f.replace('.tif', '.bmp'), cv2.imread(f))
        os.system('rm -rf ' + f)

def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '-')
    plt.show()

def random_color_distort(src, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                         saturation_low=0.5, saturation_high=1.5, hue_delta=18):
    """
    gluoncv/data/transforms/experimental/image.py
    Randomly distort image color space.
    Note that input image should in original range [0, 255].
    Parameters
    ----------
    src : numpy.ndarray
        Input image as HWC format.
    brightness_delta : int
        Maximum brightness delta. Defaults to 32.
    contrast_low : float
        Lowest contrast. Defaults to 0.5.
    contrast_high : float
        Highest contrast. Defaults to 1.5.
    saturation_low : float
        Lowest saturation. Defaults to 0.5.
    saturation_high : float
        Highest saturation. Defaults to 1.5.
    hue_delta : int
        Maximum hue delta. Defaults to 18.
    Returns
    -------
    numpy.ndarray
        Distorted image in HWC format.
    """
    def brightness(src, delta, p=0.5):
        """Brightness distortion."""
        if np.random.uniform(0, 1) > p:
            delta = np.random.uniform(-delta, delta)
            src += delta
            return src
        return src

    def contrast(src, low, high, p=0.5):
        """Contrast distortion"""
        if np.random.uniform(0, 1) > p:
            alpha = np.random.uniform(low, high)
            src *= alpha
            return src
        return src

    def saturation(src, low, high, p=0.5):
        """Saturation distortion."""
        if np.random.uniform(0, 1) > p:
            alpha = np.random.uniform(low, high)
            gray = src * np.array([[[0.299, 0.587, 0.114]]])
            gray = np.sum(gray, axis=2, keepdims=True)
            gray *= (1.0 - alpha)
            src *= alpha
            src += gray
            return src
        return src

    def hue(src, delta, p=0.5):
        """Hue distortion"""
        if np.random.uniform(0, 1) > p:
            alpha = random.uniform(-delta, delta)
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                           [0.0, u, -w],
                           [0.0, w, u]])
            tyiq = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]])
            ityiq = np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]])
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            src = np.dot(src, t)
            return src
        return src

    src = src.astype(np.float32)
    # brightness
    src = brightness(src, brightness_delta)

    # color jitter
    if np.random.randint(0, 2):
        src = contrast(src, contrast_low, contrast_high)
        src = saturation(src, saturation_low, saturation_high)
        src = hue(src, hue_delta)
    else:
        src = saturation(src, saturation_low, saturation_high)
        src = hue(src, hue_delta)
        src = contrast(src, contrast_low, contrast_high)
    # return np.clip(src, 0, 255).astype(np.uint8)
    return src

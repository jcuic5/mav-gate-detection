import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import glob, os
import numpy as np
from PIL import Image

import cv2

class GateDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
#         targets = torch.tensor(targets.copy())
#         self.targets = torch.nn.functional.one_hot(targets.to(torch.int64))
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]
#         print(img.shape)
        if self.transform:
            img = self.transform(img)
#         print(img.shape)
        
        target = self.targets[idx]

        return img, target

class GateResNet18(nn.Module):    
    def __init__(self, out_features):
        super(GateResNet18, self).__init__()
        
        self.resnet18 = models.resnet18()
        self.fc = nn.Linear(1000, out_features)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)
        
        return x

class GateSegDataset(object):
    def __init__(self, img_filenames, mask_filenames, transforms=None):
        self.transforms = transforms
        self.imgs = img_filenames
        self.masks = mask_filenames

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

from engine import train_one_epoch, evaluate
import utils
import transforms as T

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def find_good_matches(matches, queryKp, trainKp):
# FILTER GOOD MATCHES
    good_matches = []

    for candidates in matches:
        scores = np.zeros((len(candidates),))
        vert_popularity = np.zeros((len(candidates),))
        vert_similarity = np.zeros((len(candidates),))
        query_y = (queryKp[candidates[0].queryIdx].pt)[1] # NOTE: m.queryIdx == n.queryIdx
        
        for idx, candidate in enumerate(candidates):
            scores[idx] = candidate.distance
            train_cy = (trainKp[candidate.trainIdx].pt)[1]
            vert_popularity[idx] = train_cy
            vert_similarity[idx] = abs(query_y - train_cy)
        vert_popularity = abs(vert_popularity - np.median(vert_popularity))
        scores += 1 *vert_popularity + 0*vert_similarity
        
        scores = -scores
        scores -= np.min(scores)
        best_candidate = np.argmax(scores)
        best_score = scores[best_candidate]
        scores[best_candidate] = float('-inf')
        second_candidate = np.argmax(scores)
        if best_score > 1.25 * scores[second_candidate]:
            good_matches.append(candidates[best_candidate])
        
    return good_matches

MIN_MATCH_COUNT = 10

def perspective_transform(good_matches, queryKp, trainKp, queryMask):
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([queryKp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([trainKp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w = queryMask.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        
        PredMask = cv2.warpPerspective(queryMask, M, (queryMask.shape[1], queryMask.shape[0]))
        return PredMask
        
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCH_COUNT))
        matchesMask = None
        return None
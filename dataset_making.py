# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import glob, os
import pickle
import skimage
import skimage.transform
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import matplotlib.patches as patches
import cv2

import torch
from PIL import Image

def show_corners(I, points):
    plt.figure()
        # show the image
    # plot point p1 as a green circle, with markersize 10, and label "tip"
    # plot point p2 as a red circle, with markersize 10, and label "end"
    # plot a line starts at one point and end at another. 
    # Use a suitable color and linewidth for better visualization
    # Add a legend (tip, you can use the "label" keyword when you plot a point)
    
    plt.imshow(I)
    sc1 = plt.scatter(points[0], points[1], s=25, marker="o", color="magenta")
    sc2 = plt.scatter(points[2], points[3], s=25, marker="o", color="magenta")
    sc3 = plt.scatter(points[4], points[5], s=25, marker="o", color="magenta")
    sc4 = plt.scatter(points[6], points[7], s=25, marker="o", color="magenta")
    # done, show the image
    plt.show()

# PATCH SIZE
WIN_SIZE = (28, 28, 3)
HALF_WIN_SIZE = (WIN_SIZE[0] // 2, WIN_SIZE[1] // 2, WIN_SIZE[2])

def sample_points_grid(I):
    # window centers
    W = I.shape[1]
    H = I.shape[0]
    
    step_size = (WIN_SIZE[0] // 4, WIN_SIZE[1] // 4)
    min_ys = range(0, H-WIN_SIZE[0]+1, step_size[0])
    min_xs = range(0, W-WIN_SIZE[1]+1, step_size[1])
    center_ys = range(HALF_WIN_SIZE[0], H-HALF_WIN_SIZE[0]+1, step_size[0])
    center_xs = range(HALF_WIN_SIZE[1], W-HALF_WIN_SIZE[1]+1, step_size[1])
    centers = np.array(np.meshgrid(center_xs, center_ys))
    centers = centers.reshape(2,-1).T
    centers = centers.astype(float) 
    
#     # add a bit of random offset
#     centers += np.random.rand(*centers.shape) * 10
    
    # discard points close to border where we can't extract patches
    centers = remove_points_near_border(I, centers)
    
    return centers

# REMOVE SAMPLED POINTS NEAR BORDER
def remove_points_near_border(I, points):
    W = I.shape[1]
    H = I.shape[0]

    # discard points that are too close to border
    points = points[points[:,0] > HALF_WIN_SIZE[1],:]
    points = points[points[:,1] > HALF_WIN_SIZE[0],:]
    points = points[points[:,0] < W - HALF_WIN_SIZE[1],:]
    points = points[points[:,1] < H - HALF_WIN_SIZE[0],:]
    
    return points

# SAMPLE POINTS WITH UNIFORM GRIDS AND ALSO AROUND GATES(STRATEGY 2)
def sample_points_around_gates(I, corners):    
    Nu = 60 # uniform samples(will mostly be background)
    Nt = 5 # samples at target locations(corners and between)
    
    target_std_dev = np.array(HALF_WIN_SIZE[:2]) / 8 # variance to add to locations

    # uniform samples
    upoints = sample_points_grid(I)
    idxs = np.random.choice(upoints.shape[0], Nu)
    upoints = upoints[idxs,:]
    
    points = upoints
    
    gate_pts = corners.copy().reshape(-1, 8)
    for idx in range(gate_pts.shape[0]):
        pts = gate_pts[idx]
        ptss = np.concatenate((pts, pts[2:], pts[:2]))
#         print(ptss)
        for pair_idx in range(4):
            p1 = ptss[2*pair_idx:2*pair_idx+2]
            p2 = ptss[2*pair_idx+8:2*pair_idx+8+2]
#             print(p1, p2)
            
            # sample around corners
            tpoints1 = np.random.randn(int(Nt),2)
            tpoints1 = tpoints1 * target_std_dev/3 + p1

#             tpoints2 = np.random.randn(int(2*Nt),2)
#             tpoints2 = tpoints2 * target_std_dev/3 + p2

            # sample over gate between corner points
            alpha = np.random.rand(Nt)
            tpoints3 = p1[None,:] * alpha[:,None] + p2[None,:] * (1. - alpha[:,None])
            tpoints3 = tpoints3 + np.random.randn(Nt,2) * target_std_dev/3

            # merge points
            points = np.vstack((points, tpoints1, tpoints3))
            
    # discard points close to border where we can't extract patches
    points = remove_points_near_border(I, points)
    
    return points

# MAKE LABELS FOR POINTS USING MASKS
def make_labels_for_points(I, points, mask):
    row_idxs = np.floor(points[:, 1]).astype(int)
    col_idxs = np.floor(points[:, 0]).astype(int)
#     print(points.shape)
#     print(row_idxs.shape, col_idxs.shape)
#     print(mask.shape)
    labels = mask[row_idxs, col_idxs]
    labels[labels > 0] = 1
#     print(labels.shape)
    
    return labels

CLASS_NAMES = [
    'background',   # class 0
    'gate',        # class 1
]

def plot_labeled_points(points, labels):
    plt.plot(points[labels == 0, 0], points[labels == 0, 1], 'c.', label=CLASS_NAMES[0])
    plt.plot(points[labels == 1, 0], points[labels == 1, 1], 'r.', label=CLASS_NAMES[1])
#     plt.plot(points[labels == 2, 0], points[labels == 2, 1], 'b.', label=CLASS_NAMES[2])
#     plt.plot(points[labels == 3, 0], points[labels == 3, 1], 'y.', label=CLASS_NAMES[3])

# GET PATCH AROUND A SAMPLED POINT
def get_patch_at_point(I, p):
    idx = p.astype(int)
    x = idx[0]
    y = idx[1]
    P = I[y-HALF_WIN_SIZE[0]:y+HALF_WIN_SIZE[0], x-HALF_WIN_SIZE[1]:x+HALF_WIN_SIZE[1], 0:HALF_WIN_SIZE[2]]
    
    return P

FEAT_SIZE = (28,28,3)

# USE RESIZE FUNCTION TO REDUCE THE DIMENSIONALITY TO FEATURE SIZE
def patch_to_vec(P):
    x = skimage.transform.resize(P, FEAT_SIZE).reshape(-1,1)
    x = np.squeeze(x, axis = 1)
    
    return x

# SAMPLE POINTS AND EXTRACT PATCHES FOR A SINGLE IMAGE
def extract_patches(I, corners, mask, strategy=None):
    
    # by default, if no strategy is explicitly defined, use strategy 2
    if strategy == 1:
        points = sample_points_grid(I)
    if strategy == 2 or strategy is None:
        points = sample_points_around_gates(I, corners)
    
    # determine the labels of the points
    labels = make_labels_for_points(I, points, mask)
    
    xs = []
    for p in points:
        P = get_patch_at_point(I, p)
        x = patch_to_vec(P)
        xs.append(x)
    X = np.array(xs)

    return X, labels, points

def count_classes(labels):
#     counts = np.zeros((4,), dtype=int)
    counts = np.zeros((2,), dtype=int)
    u, cnts = np.unique(labels, return_counts=True)
    for idx in range(u.shape[0]):
        counts[u[idx]] = cnts[idx]
    return counts

# SAMPLE POINTS AND EXTRACT PATCHES FOR ALL IMAGES
def extract_multiple_images(idxs, Is, corners_list, masks, strategy=None):
    Xs = []
    ys = []
    points = []
    imgids = []

    for step, idx in enumerate(idxs):
        I = Is[idx]
        corners = corners_list[idx][1]
        mask = masks[idx]
        I_X, I_y, I_points = extract_patches(I, corners, mask, strategy=strategy)

        classcounts = count_classes(I_y)
        print(f'image {idx}, class count = {classcounts}')

        Xs.append(I_X)
        ys.append(I_y)
        points.append(I_points)
        imgids.append(np.ones(len(I_y),dtype=int)*idx)

    # Xs with shape(number of patches, dimension of feature vector 243)
    Xs = np.vstack(Xs)
    # ys is a vector with just one dimension so hstack horizontally append them as a long vector
    # so whatever a row or column vector, each idx represents a patch's label
    # ys with shape(number of patches, patch label)
    ys = np.hstack(ys)
    points = np.vstack(points)
    imgids = np.hstack(imgids)
    
    return Xs, ys, points, imgids

def plot_samples(Ps, labels):
    uls = np.unique(labels)
    nclasses = len(uls)
    nsamples = 12
    
    plt.figure(figsize=(10,4))
    
    for lidx, label in enumerate(uls):
        idxs = np.where(labels == label)[0]
        idxs = np.random.choice(idxs, nsamples, replace=False)
        
        for j, idx in enumerate(idxs):
            P = Ps[idx,:]
            P = P.reshape(FEAT_SIZE)
            
            plt.subplot(nclasses, nsamples, lidx*nsamples+j+1)
            plt.imshow(P, clim=(0,1))
            plt.axis('off')
            plt.title('label: %d' % label)
        
    plt.show()

def evaluate_accuracy(data_loader, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    net.eval()  #make sure network is in evaluation mode

    #init
    acc_sum = torch.tensor([0], dtype=torch.float32, device=device)
    n = 0

    for X, y in data_loader:
        # Copy the data to device.
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            X = X.float()
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0] #increases with the number of samples in the batch
    return acc_sum.item()/n
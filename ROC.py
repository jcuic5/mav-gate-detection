# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:40:14 2018

Generating an ROC curve.

@author: LocalAdmin
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_ROC_curve(values, classes):

    # get number of positives and negatives:    
    n_values = len(values);
    totalP = len(np.where(classes > 0)[0]);
    totalN = n_values - totalP;
    
    # sort all values:
    inds = np.argsort(values);
    s_values = values[inds];
    s_classes = classes[inds];

    TP = np.zeros([n_values, 1]);
    FP = np.zeros([n_values, 1]);

    for e in range(n_values):
        # threshold = s_values[e]
        # Positive when bigger:
        P = np.sum(s_classes[e:]);
        TP[e] = P / totalP;
        
        # number of false positives is the remaining samples above the
        # threshold divided by all negative samples:
        FP[e] = (len(s_classes[e:]) - P) / totalN;

    return TP, FP;


def get_images_and_grid(im_name, segmentation_name, show_images = True):
    
    # read RGB image:
    Im = cv2.imread(im_name);
    RGB = np.copy(Im);
    RGB[:,:,0] = Im[:,:,2];
    RGB[:,:,2] = Im[:,:,0];
    WIDTH = Im.shape[1];
    print(WIDTH)
    HEIGHT = Im.shape[0];
    print(HEIGHT)
    if(show_images):
        plt.figure();
        plt.imshow(RGB);
        plt.title('RGB Image');
        #cv2.imshow('Image', Im);
        #cv2.waitKey();
    
    # read Classification image:
    Cl = cv2.imread(segmentation_name);
    if(show_images):
        plt.figure();
        plt.imshow(Cl);
        plt.title('Classification');
        #cv2.imshow('Segmentation', Cl);
        #cv2.waitKey();
    Cl = cv2.cvtColor(Cl, cv2.COLOR_BGR2GRAY);
    Cl = Cl.flatten();
    Cl = Cl > 0;
    Cl = Cl.astype(float);
    
    # make a meshgrid:
    x, y = np.meshgrid(range(WIDTH), range(HEIGHT));
    print(x, x.shape)
    print(y, x.shape)
    x = x.flatten();
    y = y.flatten();    
    
    return RGB, Cl, x, y;

def ROC_exercise(im_name, segmentation_name, show_images = True):
    
    # read RGB image:
    Im = cv2.imread(im_name);
    WIDTH = Im.shape[1];
    HEIGHT = Im.shape[0];
    if(show_images):
        cv2.imshow('Image', Im);
        cv2.waitKey();
    
    # read Classification image:
    Cl = cv2.imread(segmentation_name);
    if(show_images):
        cv2.imshow('Segmentation', Cl);
        cv2.waitKey();
    Cl = cv2.cvtColor(Cl, cv2.COLOR_BGR2GRAY);
    Cl = Cl.flatten();
    Cl = Cl > 0;
    Cl = Cl.astype(float);
    
    # make a meshgrid:
    x, y = np.meshgrid(range(WIDTH), range(HEIGHT));
    x = x.flatten();
    y = y.flatten();
    
    # take the blue channel, cv2 represents images as BGR:
    # TODO: play with the next expression:
    Values = -y #Im[:,:,0];#2];
    Values = Values.flatten();

    
    # get the ROC curve:
    TP, FP = get_ROC_curve(Values, Cl);

    plt.figure();
    plt.plot(FP, TP, 'b');
    plt.xlabel('TP');
    plt.ylabel('FP');

if __name__ == '__main__':
    im_name = 'CroppedImage.bmp';
    segmentation_name = 'TreeSegmentation.bmp';
    ROC_exercise(im_name, segmentation_name);

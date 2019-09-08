'''
Shift Map Analysis for Image Restoration
Reference: Correcction of Distorted Underwater Images using Shift Map Analysis

Available datasets:
- mp4:
    Knife.mp4
    Heater1.mp4
    Heater2.mp4
    Pool1.mp4
    Pool2.mp4
    Pool3.mp4
- mat:
    expdata_brick.mat
    expdata_checkboard.mat
    middleFonts2data.mat
    expdata_large.mat
    expdata_small.mat
    expdata_tiny.mat
'''

from Functions import loadColorImage, processStack
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

def main(argv):    
    Directory = "Datasets/"
    fileName = argv
    print(Directory+fileName)
    
    tic = time.time()
    framesR, framesG, framesB = loadColorImage(Directory+fileName)
    frames_x, frames_y, frames_z = np.shape(framesR)
    original = np.zeros((frames_x,frames_y,3), "uint8")
    output = np.zeros((frames_x,frames_y,3), "uint8")
    
    original[:,:,0] = framesR[:,:,0]*255
    original[:,:,1] = framesG[:,:,0]*255
    original[:,:,2] = framesB[:,:,0]*255
    original = Image.fromarray(original)
    
    output[:,:,0] = processStack(framesR)*255
    output[:,:,1] = processStack(framesG)*255
    output[:,:,2] = processStack(framesB)*255
    output = Image.fromarray(output)
     
    toc = time.time()       
    print("Processing time", toc-tic, "seconds")
    plt.subplot(1,2,1)
    plt.imshow(original, cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(output, cmap="gray")
    plt.show()     

import sys 
if __name__ == "__main__":
    main(sys.argv[1])
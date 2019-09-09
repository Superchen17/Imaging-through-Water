'''
Shift Map Analysis for Image Restoration
Reference: Correcction of Distorted Underwater Images using Shift Map Analysis

Available datasets:
- mp4:
    Knife.mp4 (colo/mono)
    Heater1.mp4 (colo/mono)
    Heater2.mp4 (colo/mono)
    Pool1.mp4 (colo/mono)
    Pool2.mp4 (colo/mono)
    Pool3.mp4 (colo/mono)
- mat:
    expdata_brick.mat (mono)
    expdata_checkboard.mat (mono)
    middleFonts2data.mat (mono)
    expdata_large.mat (mono)
    expdata_small.mat (mono)
    expdata_tiny.mat (mono)
'''

from Functions import loadColorImage, loadMonoImage, processStack
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
from scipy.io import savemat

def main(argv):    
    Directory = "Datasets/"
    fileName = argv
    print(Directory+fileName)
    
    tic = time.time()
    framesR, framesG, framesB = loadColorImage(Directory+fileName)
    template  = loadMonoImage(Directory+fileName)
    frames_x, frames_y, frames_z = np.shape(framesR)
    original = np.zeros((frames_x,frames_y,3), "uint8")
    output = np.zeros((frames_x,frames_y,3), "uint8")
    
    original[:,:,0] = framesR[:,:,0]*255
    original[:,:,1] = framesG[:,:,0]*255
    original[:,:,2] = framesB[:,:,0]*255
    original = Image.fromarray(original)
    
    output[:,:,0] = processStack(framesR, template, 'color')*255 # this is to be parallelised
    output[:,:,1] = processStack(framesG, template, 'color')*255 # this is to be parallelised
    output[:,:,2] = processStack(framesB, template, 'color')*255 # this is to be parallelised
    
    savemat(str(fileName)+'_Color.mat', {'recon':output})

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
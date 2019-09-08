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

#Main method 
from Functions import loadMonoImage, processStack
import time
import matplotlib.pyplot as plt

def main(argv):    
    Directory = "Datasets/"
    fileName = argv
    print(Directory+fileName)
    
    tic = time.time()
    frames = loadMonoImage(Directory+fileName)  
    original = frames[:,:,0]

    output = processStack(frames)
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
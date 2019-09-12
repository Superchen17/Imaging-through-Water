'''
Function class
'''

'''
Imports
'''
import cv2
import scipy
import numpy as np
import pywt
import math
import queue
from scipy.io import loadmat
from skimage.transform import rescale, resize, downscale_local_mean

'''
Grayscale image loader
'''
def loadMonoImage(fileName):
    frames = []
    frames_x = 0
    frames_y = 0
    frames_z = 0    
    pixelMultiplier = 255
    upscaler = 1
    rescaler = 0.25    
    
    # Reading MATLAB .mat file
    if(fileName.endswith(".mat")):
        frames = loadmat(fileName, appendmat=False).get('frames')   
        frames = frames[:,:,0:60]*pixelMultiplier 
        image = rescale(frames[:,:,0], upscaler, anti_aliasing=True)    
        frames_new = np.zeros((np.shape(image)[0],np.shape(image)[1],np.shape(frames)[2]))
        for i in range(np.shape(frames)[2]):
            image = rescale(frames[:,:,i], upscaler, anti_aliasing=True)
            frames_new[:,:,i] = image        
        frames = frames_new  
        
    # Reading .mp4 file  
    elif(fileName.endswith(".mp4")):
        cap = cv2.VideoCapture(fileName)        
        frames_x = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_y = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames_z = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if(frames_z > 60):
            frames_z = 60
        frames = np.zeros((frames_x,frames_y,frames_z))
        fc = 0
        ret = True
        while (fc < frames_z  and ret):
            ret, frame = cap.read()
            frames[:,:,fc] = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            fc += 1   
        cap.release()  
        
        frames = resize(frames,(frames_x//4,frames_y//4), anti_aliasing=True)
    
    frames_x,frames_y,frames_z = np.shape(frames) 
    print("Image loaded:", frames_x, frames_y, frames_z)   
    return frames

'''
Color image loader
'''
def loadColorImage(fileName):    
    framesR = [];
    framesG = [];
    framesB = [];
    frames_x = 0;
    frames_y = 0;
    frames_z = 0;
    rescaler = 0.25
    
    cap = cv2.VideoCapture(fileName)        
    frames_x = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_y = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames_z = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if(frames_z > 60):
        frames_z = 60    
    framesR = np.zeros((frames_x,frames_y,frames_z))
    framesG = np.zeros((frames_x,frames_y,frames_z))
    framesB = np.zeros((frames_x,frames_y,frames_z))    
    
    fc = 0
    ret = True
    while (fc < frames_z  and ret):
        ret, frame = cap.read()
        framesB[:,:,fc] = frame[:,:,0]
        framesG[:,:,fc] = frame[:,:,1]
        framesR[:,:,fc] = frame[:,:,2]       
        fc += 1   
    cap.release()     

    framesR = resize(framesR,(frames_x//4,frames_y//4), anti_aliasing=True)
    framesG = resize(framesG,(frames_x//4,frames_y//4), anti_aliasing=True)
    framesB = resize(framesB,(frames_x//4,frames_y//4), anti_aliasing=True)  
    
    frames_x,frames_y,frames_z = np.shape(framesG) 
    print("Image loaded:", frames_x, frames_y, frames_z)   
    return framesR, framesG, framesB

'''
Gap filling for forward mapping
'''
def denoise(input): 
    input = np.float32(input)
    ret, mask = cv2.threshold(input,1,255, cv2.THRESH_BINARY_INV)
    mask = mask.astype('uint8')
    input = cv2.inpaint(input,mask,3,cv2.INPAINT_TELEA)     
    return input 

'''
Forward mapping according to warping field
'''
def forWardDewarping(input, shiftRuleX, shiftRuleY):
    frames_x,frames_y = np.shape(input)
    output = np.zeros_like(input)
    for i in range (frames_x):
        for j in range (frames_y):
            xNew = np.int0(i+shiftRuleX[i,j])
            yNew = np.int0(j+shiftRuleY[i,j])
            if(xNew >= frames_x):
                xNew = frames_x-1
            if(yNew >= frames_y):
                yNew = frames_y-1
            if(xNew < 0):
                xNew = 0
            if(yNew < 0):
                yNew = 0    
            output[xNew,yNew] = input[i,j]
    output = denoise(output)
    return output

'''
Backward mapping according to warping field
'''
def backDewarping(input, shiftRuleX, shiftRuleY):
    frames_x, frames_y = np.shape(input)
    output = np.zeros_like(input)
    for i in range (frames_x):
        for j in range (frames_y):
            xNew = np.int0(i+shiftRuleX[i,j])
            yNew = np.int0(j+shiftRuleY[i,j])
            if(xNew >= frames_x):
                xNew = frames_x-1
            if(yNew >= frames_y):
                yNew = frames_y-1
            if(xNew < 0):
                xNew = 0
            if(yNew < 0):
                yNew = 0    
            output[i,j] = input[xNew,yNew]
    return output 

'''
Wavelet coefficient fusion rule
'''
def fuseCoeff(cooef1, cooef2, method):
    if (method == 'mean'):
        cooef = (cooef1 + cooef2)/2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    elif (method == 'add'):
        cooef = cooef1+cooef2
        cooef = cooef*max(np.max(cooef1),np.max(cooef2))/np.max(cooef) 
    elif (method == 'merge'):
        cooef = sharpFuse(cooef1,cooef2)    
    else:
        cooef = []
    return cooef

'''
Image fusion with wavelet coefficient decomposition
'''
def waveletFuse(input1,input2):
    ## Fusion algo   
    # First: Do wavelet transform on each image
    wavelet = 'bior1.1'
    cooef1 = pywt.wavedec2(input1, wavelet)
    cooef2 = pywt.wavedec2(input2, wavelet)
    
    # Second: for each level in both image do the fusion according to the desire option
    fusedCooef = []
    for i in range(len(cooef1)):    
        # The first values in each decomposition is the apprximation values of the top level      
        if(i == 0):  
            fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],'mean')) #LL            
        else:
            # For the rest of the levels we have tupels with 3 coefficents                
            c1 = fuseCoeff(cooef1[i][0],cooef2[i][0],'mean') #LH 
            c2 = fuseCoeff(cooef1[i][1],cooef2[i][1],'mean') #HL
            c3 = fuseCoeff(cooef1[i][2],cooef2[i][2],'mean') #HH                                              
            fusedCooef.append((c1,c2,c3))                
    # Third: After we fused the cooefficent we need to transfor back to get the image
    fusedImage = pywt.waverec2(fusedCooef, wavelet)        
    return fusedImage

'''
Quality map generation 
M = J*G
'''
from scipy.ndimage import gaussian_filter  
def getQualityMap(input):
    global frames_x, frames_y
    sig = np.std(input)   
    grad_x = cv2.Sobel(input,cv2.CV_64F,1,0,ksize=3)
    grad_y = cv2.Sobel(input,cv2.CV_64F,0,1,ksize=3)
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    sobel = (2*math.pi*(sig**2))*gaussian_filter(grad, sigma=sig)          
    return sobel 

'''
Image fusion with sharpness analysis
'''
def sharpFuse(input1, input2):
    frames_x,frames_y = np.shape(input1)
    quality1 = getQualityMap(input1)
    quality2 = getQualityMap(input2)
    deltaMap = np.zeros_like(input1)
    for x in range(frames_x):
        for y in range(frames_y):
            if(quality1[x,y] > quality2[x,y]):
                deltaMap[x,y] = quality1[x,y]-quality2[x,y]
    if(np.max(deltaMap != 0)):
        ka = 1/np.max(deltaMap)
        output = (1-ka*deltaMap)*input2+ka*deltaMap*input1 
    else:
        output = input1
    return output
    
'''
Adjacent image fusion procedure
'''
def processStack(frames):
    frames_x, frames_y, frames_z = np.shape(frames)
    
    while(frames_z>1):
        frames_new = np.zeros((frames_x,frames_y,math.ceil(frames_z/2)))
        for i in range(0, frames_z-1,2):
            print("Iteration", i, "/", frames_z-1)
            reference = frames[:,:,i]
            target = frames[:,:,i+1]
            
            flow_rt = cv2.calcOpticalFlowFarneback(target,reference,None,0.5,3,15,3,5,1.2,0)/2
            warped_rt = backDewarping(reference, flow_rt[:,:,1], flow_rt[:,:,0])
            flow_tr = cv2.calcOpticalFlowFarneback(reference,target,None,0.5,3,15,3,5,1.2,0)/2                                    
            warped_tr = backDewarping(target, flow_tr[:,:,1], flow_tr[:,:,0]) 
            
            warped = waveletFuse(warped_rt,warped_tr)
            warped = cv2.resize(warped,(frames_y,frames_x))  
            frames_new[:,:,i//2] = warped
            
            if(i == frames_z-3):
                frames_new[:,:,(i//2)+1] = frames[:,:,-1]
            
        frames = frames_new
        frames_x, frames_y, frames_z = np.shape(frames)        
    output = frames[:,:,0]
   
    return output


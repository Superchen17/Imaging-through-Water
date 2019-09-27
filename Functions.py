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
from multiprocessing import Pool
from numba import jit

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
    rescaler = 4
    
    # Reading MATLAB .mat file
    if(fileName.endswith(".mat")):
        frames = loadmat(fileName, appendmat=False).get('frames')   
        frames = frames[:,:,0:60]*pixelMultiplier 
        frames_x,frames_y,frames_z = np.shape(frames)         
        frames = cv2.resize(frames,(frames_y*upscaler,frames_x*upscaler), interpolation=cv2.INTER_CUBIC)         
        
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
        
        frames = resize(frames,(frames_x//rescaler,frames_y//rescaler), anti_aliasing=True)
    
    frames_x,frames_y,frames_z = np.shape(frames) 
    print("Image loaded:", frames_x, frames_y, frames_z)   
    return frames

'''
Color image loader
'''
def loadColorImage(fileName):    
    framesR = []
    framesG = []
    framesB = []
    frames_x = 0
    frames_y = 0
    frames_z = 0
    rescaler = 4
    
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

    framesR = resize(framesR,(frames_x//rescaler,frames_y//rescaler), anti_aliasing=True)
    framesG = resize(framesG,(frames_x//rescaler,frames_y//rescaler), anti_aliasing=True)
    framesB = resize(framesB,(frames_x//rescaler,frames_y//rescaler), anti_aliasing=True)  
    
    frames_x,frames_y,frames_z = np.shape(framesG) 
    print("Image loaded:", frames_x, frames_y, frames_z)   
    return framesR, framesG, framesB

'''
Gap filling for forward mapping
'''
@jit
def denoise(input): 
    input = np.float32(input)
    ret, mask = cv2.threshold(input,1,255, cv2.THRESH_BINARY_INV)
    mask = mask.astype('uint8')
    input = cv2.inpaint(input,mask,3,cv2.INPAINT_TELEA)     
    return input 

'''
Forward mapping according to warping field, not used
'''
@jit
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
    return output

'''
Backward mapping accelerated with CUDA
'''
'''
@jit
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
'''
@jit
def backDewarping(input, shiftRuleX, shiftRuleY):
    frames_x_ori, frames_y_ori = np.shape(input)
    upscalar = 2
    input = cv2.resize(input,(frames_y_ori*upscalar,frames_x_ori*upscalar), interpolation=cv2.INTER_CUBIC) 
    shiftRuleX = cv2.resize(shiftRuleX,(frames_y_ori*upscalar,frames_x_ori*upscalar), interpolation=cv2.INTER_CUBIC)*upscalar 
    shiftRuleY = cv2.resize(shiftRuleY,(frames_y_ori*upscalar,frames_x_ori*upscalar), interpolation=cv2.INTER_CUBIC)*upscalar 
        
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
            
    output = cv2.resize(output,(frames_y_ori,frames_x_ori),interpolation=cv2.INTER_CUBIC)     
    return output 
'''

@jit
def backDewarping(input, shiftRuleX, shiftRuleY):
    frames_x_ori, frames_y_ori = np.shape(input)
    upscalar = 2
    input = cv2.resize(input,(frames_y_ori*upscalar,frames_x_ori*upscalar), interpolation=cv2.INTER_CUBIC) 
    shiftRuleX = cv2.resize(shiftRuleX,(frames_y_ori*upscalar,frames_x_ori*upscalar), interpolation=cv2.INTER_CUBIC)*upscalar 
    shiftRuleY = cv2.resize(shiftRuleY,(frames_y_ori*upscalar,frames_x_ori*upscalar), interpolation=cv2.INTER_CUBIC)*upscalar
    mx = np.zeros_like(shiftRuleX)
    my = np.zeros_like(shiftRuleY)
        
    frames_x, frames_y = np.shape(input)    
    output = np.zeros_like(input)
    
    for i in range (frames_x):
        for j in range (frames_y):
            mx[i,j] = i+shiftRuleX[i,j]
            my[i,j] = j+shiftRuleY[i,j]
            
    output = cv2.remap(input,my,mx,interpolation=cv2.INTER_CUBIC,borderMode=2)
    output = cv2.resize(output,(frames_y_ori,frames_x_ori),interpolation=cv2.INTER_CUBIC)     
    return output 

'''
Wavelet coefficient fusion rule
'''
def fuseCoeff(cooef1, cooef2, method):
    lengthX, lengthY = np.shape(cooef1)
    cooef = []    
    if (method == 'mean'):
        cooef = (cooef1 + cooef2)/2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    elif (method == 'merge'):
        #cooef = PCAFuse(cooef1,cooef2) 
        weight = getPCAWeight(cooef1,cooef2)
        cooef = weight*cooef1+(1-weight)*cooef2
    return cooef

'''
Image fusion with wavelet coefficient decomposition
'''
def waveletFuse(input1,input2):
    ## Fusion algo   
    # First: Do wavelet transform on each image
    wavelet = 'haar'
    cooef1 = pywt.wavedec2(input1, wavelet)
    cooef2 = pywt.wavedec2(input2, wavelet)
    
    policy = 'merge'        
    # Second: for each level in both image do the fusion according to the desire option
    fusedCooef = []
    for i in range(len(cooef1)):    
        # The first values in each decomposition is the apprximation values of the top level  
        if(i == 0):  
            fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],policy)) #LL            
        else:
            # For the rest of the levels we have tupels with 3 coefficents                
            c1 = fuseCoeff(cooef1[i][0],cooef2[i][0],policy) #LH 
            c2 = fuseCoeff(cooef1[i][1],cooef2[i][1],policy) #HL
            c3 = fuseCoeff(cooef1[i][2],cooef2[i][2],policy) #HH                                              
            fusedCooef.append((c1,c2,c3))                
    # Third: After we fused the cooefficent we need to transfor back to get the image
    fusedImage = pywt.waverec2(fusedCooef, wavelet)        
    return fusedImage

'''
Get weight of images with PCA
'''
def getPCAWeight(input1,input2):
    img1 = input1.flatten()
    img2 = input2.flatten()    
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)
    
    img = []
    img.append(img1)
    img.append(img2)

    cov = np.cov(img)
    w,v = scipy.linalg.eigh(cov)
    idx = np.argsort(w)
    idx = idx[::-1]
    v = v[:,idx]
    w = w[idx]    
    v_max = np.abs(v[:,0])
   
    p1 = v_max[0]/np.sum(v_max)   
    return p1

'''
Quality map generation, not used
M = J*G
'''
from scipy.ndimage import gaussian_filter  
from sklearn.preprocessing import normalize
def getQualityMap(input):
    frames_x,frames_y = np.shape(input)
    grad_x = cv2.Sobel(input,cv2.CV_64F,1,0,ksize=3)
    grad_y = cv2.Sobel(input,cv2.CV_64F,0,1,ksize=3)
    grad = np.hypot(grad_x,grad_y)/np.sum(input)
    grad = cv2.GaussianBlur(grad,(5,5),0)
    return grad 

'''
Image fusion with sharpness analysis, not used
'''
@jit
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
Model
'''
def processStack(frames):
    frames_x, frames_y, frames_z = np.shape(frames)  
    
    counter = 1
    maxLevel = math.ceil(math.log2(60))
    while(frames_z>1):
        print("Iteration", counter, "/", maxLevel)
        frames_new = np.zeros((frames_x,frames_y,math.ceil(frames_z/2)))
        for i in range(0, frames_z-1,2):
            reference = frames[:,:,i]
            target = frames[:,:,i+1] 
            
            flow_rt = cv2.calcOpticalFlowFarneback(target,reference,None,0.5,3,13,3,7,1.5,0)/2
            flow_tr = cv2.calcOpticalFlowFarneback(reference,target,None,0.5,3,13,3,7,1.5,0)/2               
            warped_rt = backDewarping(reference, flow_rt[:,:,1], flow_rt[:,:,0])            
            warped_tr = backDewarping(target, flow_tr[:,:,1], flow_tr[:,:,0]) 
            
            warped = waveletFuse(warped_rt,warped_tr)
            warped = cv2.resize(warped,(frames_y,frames_x), interpolation=cv2.INTER_CUBIC) 
            
            frames_new[:,:,i//2] = warped

            if(i == frames_z-3):
                frames_new[:,:,(i//2)+1] = frames[:,:,-1]
            
        frames = frames_new
        frames_x, frames_y, frames_z = np.shape(frames)   
        counter += 1
    output = frames[:,:,0]
   
    return output


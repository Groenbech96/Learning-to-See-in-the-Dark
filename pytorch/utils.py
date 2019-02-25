import glob
import os
import rawpy
import numpy as np

# Returns a tuble 
def getInputImagesList():
    # Get all short exposure images
    res = glob.glob('./dataset/sony/short/0*.ARW')
    res.sort()
    return (res, [int(os.path.basename(res)[0:5]) for res in res])

def getGroundtruthImagesList():
    # Get all short exposure images
    res = glob.glob('./dataset/sony/long/0*.ARW')
    res.sort()
    return (res, [int(os.path.basename(res)[0:5]) for res in res])

def getTestInputImagesList():
    res = glob.glob('./dataset/sony/short/1*.ARW')
    res.sort()

    return(res, [int(os.path.basename(res)[0:5]) for test_fn in test_fns])

def getTestGroundtruthImagesList():
    # Get all short exposure images
    res = glob.glob('./dataset/sony/long/1*.ARW')
    res.sort()
    return (res, [int(os.path.basename(res)[0:5]) for res in res])

# Use rawpy to get pictures
def pack_raw(raw):
    # Pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)

    # Subtract the black level
    # 16383 == 2^14 (data is 14 bits)
    # 512 is hardware specific to the camera 
    im = np.maximum(im - 512, 0) / (16383 - 512)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def pack_raw_test(raw):
    #pack Bayer image to 4 channels
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out
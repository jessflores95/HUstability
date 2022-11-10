import os
import glob
import pydicom
import numpy as np
from scipy.stats import norm


def createVol(ima_dir): # input is directory containing slices
    dir_ = os.path.join(ima_dir, '*.dcm')
    files = []
    for fname in glob.glob(dir_, recursive=False):
        files.append(pydicom.dcmread(fname))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1


    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    vx_size = np.array((ss, ps[0], ps[1]))
    print(vx_size)

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d
    print(slices[0].ConvolutionKernel)
    slope = slices[0].RescaleSlope
    intercept = slices[0].RescaleIntercept
    img3d = slope*img3d + intercept
    img3d = img3d.transpose((2, 0, 1))     
    img_dict = {'img':img3d}
        
    return img_dict
    
    
def createCircularMask(img, center, radius):  # create circular mask
    # center is the coordinate in voxels of the center of desired ROI
    _, y, x = img.shape
    Y, X = np.ogrid[:y, :x]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = dist_from_center <= radius
    return mask

def meanROIvalue(ROI_dict, img_dict):
    img = img_dict['img']
    img_shape = img.shape
    mask = np.zeros((img_shape[1], img_shape[2]))
    for i in range(len(ROI_dict['insert'])):
        y, x = ROI_dict['center'][i]
        r = ROI_dict['radius'][i]
        mask = createCircularMask(img, (y, x), r)
        img_pm = img * mask
        HU_val = np.zeros(img_pm.shape[0]*mask[mask==1].shape[0])
        for k in range(img_pm.shape[0]):
            HU_val[k*mask[mask==1].shape[0]:(k+1)*mask[mask==1].shape[0]] = img_pm[k,:,:][mask==1]
        img_avg = np.mean(img_pm, axis = 0, dtype=np.float64)
        ROI_dict['mean'][i] = np.mean(HU_val, dtype=np.float64)
        ROI_dict['std'][i] = np.std(HU_val, dtype=np.float64) 
    return 
    
def coeff_var(mu, sigma):
    return (sigma/mu)*100
      
def fit_gauss(data):
    (mu, sigma) = norm.fit(data)
    fullWidth = 2*sigma*np.sqrt(2*np.log(2))
    return mu, sigma, fullWidth

def conf_interval(data):
    confidence = 0.95
    values = [np.random.choice(data,size=len(data),replace=True).mean() for i in range(1000)] 
    CI = np.percentile(values,[100*(1-confidence)/2,100*(1-(1-confidence)/2)]) 

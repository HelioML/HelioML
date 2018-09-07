
# import matplotlib as mpl
# mpl.use('Agg')

import numpy as np
# import scipy.interpolate
# import scipy.ndimage as nd
# import glob
# import matplotlib.pyplot as plt
# from congrid import resample
# import random
# from pyiacsun.util import progressbar
# import h5py
import radialProfile 
# import time
# time0 = time.time()
from astropy.convolution import AiryDisk2DKernel
# from  scipy.io import readsav
from scipy import fftpack


# def airyR(x,R):
#     Rz = 1.21966989
#     x = (np.pi*x)/(R/Rz)
#     return (2*sp.j1(x)/(x))**2.

def scatteringR(x,R):
    # http://jsoc.stanford.edu/relevant_papers/Wachter_imageQ.pdf
    Rz = 1.21966989
    x = (np.pi*x)/(R/Rz)
    e = 0.1
    k = 3.0
    w = 1.8
    W = 3.0
    return (1.-e)*np.exp(-(x/w)**2.) + e/(1.+(x/W)**k)

def scatteringR2(x,R):
    # http://jsoc.stanford.edu/relevant_papers/Wachter_imageQ.pdf
    Rz = 1.21966989
    x = (np.pi*x)/(R/Rz)
    e = 0.1
    k = 3.0
    w = 1.8*1.2
    W = 3.0*0.5
    return (1.-e)*np.exp(-(x/w)**2.) + e/(1.+(x/W)**k)

def createPSFScattering(radio):
    # radio = radioArc/(0.504302/2.)
    psfs0 = AiryDisk2DKernel(radio)

    psfs1 = np.copy(psfs0)
    x0, y0 = psfs0.center
    for ypos in range(psfs1.shape[0]):
        for xpos in range(psfs1.shape[1]):
            psfs1[ypos,xpos] = scatteringR(np.sqrt(abs(xpos-x0)**2.+abs(ypos-y0)**2.),radio)
    psfs1 /= np.sum(psfs1)
    return psfs1

# def createPSFAiry(radio):
#     # radio = radioArc/(0.504302/2.)
#     psf0 = AiryDisk2DKernel(radio)

#     psf1 = np.copy(psf0)
#     x0, y0 = psf0.center
#     for ypos in range(psf1.shape[0]):
#         for xpos in range(psf1.shape[1]):
#             # print(xpos,ypos)
#             psf1[ypos,xpos] = airyR(np.sqrt(abs(xpos-x0)**2.+abs(ypos-y0)**2.),radio)
#             # print(xpos,ypos)
#     psf1 /= np.sum(psf1)
#     return psf1

def fft1D(imagen):
    nimage = (imagen- np.median(imagen))/np.std(imagen)
    nimage = nimage.astype(float)

    nsize = 3
    image = np.zeros((nimage.shape[0]*nsize,nimage.shape[1]*nsize))
    image[0:nimage.shape[0],0:nimage.shape[1]] = nimage
    # image[nimage.shape[0]:,0:nimage.shape[1]] = np.rot90(nimage)
    # image[nimage.shape[0]:,nimage.shape[1]:] = np.rot90(np.rot90(nimage))
    # image[0:nimage.shape[0],nimage.shape[1]:] = np.rot90(np.rot90(np.rot90(nimage)))

    T = float(image.shape[0])

    # Take the fourier transform of the image.
    F1 = fftpack.fft2(image)
    F2 = fftpack.fftshift(F1)

    # Calculate a 2D power spectrum
    psf2D = np.abs(F2)**2.

    # Calculate the azimuthally averaged 1D power spectrum
    psf1D = radialProfile.azimuthalAverage(psf2D, center=(int(T/2), int(T/2)))
    v = np.arange(len(psf1D))/T
    vmax = psf2D.shape[0]/2./T
    ii = list(v).index(vmax)
    return v[1:ii], psf1D[1:ii]


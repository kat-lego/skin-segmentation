#########################################################
#                SOME LIBRARY IMPORTS                   #
#########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d as mplot3d

import skimage as sk
import skimage.feature as skfeature
import skimage.filters as skfilters
import skimage.segmentation as sksegmentation 
import skimage.color as skcolor
import skimage.measure as skmeasure
import skimage.morphology as skmorph
import skimage.exposure as skexposure

import sklearn.cluster as sk2cluster

#########################################################
#                  UTILITY FUNCTIONS                    #
#########################################################
def rgb2ycbcr(im):
    """
    Converts rgb images to ycbcr
    """
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def read_image(index, dir_t):
    """
    read image.jpg from data/dir_t directory
    """
    
    filename = "data/{t}/{i}.jpg".format(t = dir_t, i=index)    
    img = plt.imread(filename)
    return img


#########################################################
#                  THE SFA DATASET                      #
#########################################################
def sfa_dataset(dir_t, n_images = 3354):
    d = 35**2
    colors = np.zeros((n_images*d, 3));

    for i in range(0,n_images):
        tuna = read_image(i+1, dir_t).reshape(d,3)
        j,k = i*d, i*d+d
        colors[j:k] = tuna
    
    colors = rgb2ycbcr(colors.reshape(n_images,d,3)).reshape(n_images*d, 3)[:,1:]

    return colors

#########################################################
#                   THE DISTRIBUTION                    #
#########################################################
def get_mean(data):
     return np.mean(data, axis=0)

def get_covariance(data):
    return np.cov(data.T)

def likelihoods(x, mean, cov):
    x = x-mean
    
    X = np.matmul(x[:], np.linalg.inv(cov))
    X = np.sum(X*x, axis = 1)
    
    return np.exp(-0.5*X)
    
def skl_image(img , mean, cov):
    r,c,d = img.shape
    x = img.reshape(r*c,d)[:,:mean.shape[0]]
    
    L = likelihoods(x, mean , cov)
    
    return L.reshape(r,c)

#########################################################
#              UNSUPERVISED SEGMENTATION                #
#########################################################

def kmeans(image, k):
    """
    kmeans segmentation
    """
    dx,dy,dz= image.shape
    data = np.zeros((dx*dy,dz))
    data = image.reshape(dx*dy,dz)
    
    kmeans = sk2cluster.KMeans(n_clusters=k, max_iter = 1000).fit(data)
    plabels = kmeans.predict(data)
    pimg = kmeans.cluster_centers_[plabels].reshape(dx,dy,dz).astype(int)
    
    return pimg, k

def slic(image, k):
    """
    superpixel segmentation
    """
    out = sksegmentation.slic(image, n_segments=k, compactness=10, sigma=3, multichannel = True)
    k = np.unique(out).size
    out = skcolor.label2rgb(out, image, kind='avg')
    
    return out,k

#########################################################
#                   SKIN SEGMENTATION                   #
#########################################################

def skin_segmentation(img, mean, cov):
    img = rgb2ycbcr(img)

    # apply skin color model
    img = skl_image(img[:,:,1:], mean, cov)
    img = skfilters.gaussian(img, sigma=2)
    img = skmorph.opening(img, skmorph.disk(3))

    # thresholding
    thresh = skfilters.threshold_otsu(img)
    img = skmorph.closing(img > thresh, skmorph.square(3))

    # add edges to sharppern things up
    img += skfeature.canny(img, sigma=3)
    
    return img

def skin_segmentation_final(img, seg = None, segK = 10):
    res = img.copy()
    
    if seg != None:
        res,_ = seg(img, segK)
    
    SFA_DATASET = sfa_dataset('skin')
    
    mean = get_mean(SFA_DATASET)
    cov = get_covariance(SFA_DATASET)
    res = skin_segmentation(res, mean, cov)
    
    res,_ = postprocessing(res)
    
    return res

#########################################################
#                    POSTPROCESSING                     #
#########################################################

def postprocessing(img):
    
    # find the connected components
    plabels = skmorph.label(img)
     
    # region cuptoff heuristic
    areas = []
    for region in skmeasure.regionprops(plabels):
        areas.append(region.area)
    
    area_cutoff = np.max(np.array(areas))*0.05
    
    #remove small components
    for region in skmeasure.regionprops(plabels):
        if region.area <= area_cutoff:
            minr, minc, maxr, maxc = region.bbox
            img[minr:maxr,minc:maxc] = 0
    
    # border patches
    rects = []
    for region in skmeasure.regionprops(plabels):
        if region.area >= area_cutoff:
            minr, minc, maxr, maxc = region.bbox
            rects.append(
                mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                   fill=False, edgecolor='red', linewidth=2)
            )
    
    return img, rects



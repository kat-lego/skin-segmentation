import skin_segmentation as code

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d as mplot3d

Transparency = True #transparency for saved images

#########################################################
#                       DEMO 1                          #
#########################################################
def demo1(data):
    mean = code.get_mean(data)
    cov = code.get_covariance(data)
    
    Cb = np.arange(256)
    Cr = np.arange(256)
    Cb, Cr = np.meshgrid(Cb, Cr)
    
    X = np.zeros((256,256, 2))
    X[:,:,0] = Cb
    X[:,:,1] = Cr
    
    L = code.skl_image(X, mean, cov)
    
    fig = plt.figure(figsize = (9,6))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(Cb, Cr, L, linewidth=0, antialiased=True)
    ax.set_xlabel('Cb values')
    ax.set_ylabel('Cr values')
    ax.set_zlabel('P values')
    
    plt.savefig('results/demo1.png', transparent = Transparency)

    
#########################################################
#                       DEMO 2                          #
#########################################################
def demo2():
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (7*2,7*2))

    img1 = code.read_image(1, 'test')
    img2,_ = code.kmeans(img1, 30)
    img3,k = code.slic(img1, 1000)
    imgs = [img1, img2, img3]
    
    titles = [
        'a) orginal',
        'b) kmeans segmented with 25 clusters',
        'c) superpixels segmented with '+str(k)+' clusters'
    ]
    
    for j in range(3):
        axes[j].axis("off")
        axes[j].imshow(imgs[j])
        axes[j].set_title(titles[j])
    
    plt.savefig('results/demo2.png', transparent = Transparency)
    fig.tight_layout()
    
    return imgs


#########################################################
#                       DEMO 3                          #
#########################################################
def demo3(images, data):
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (7*2,8))
    
    imgs = [None, None, None]
    
    mean = code.get_mean(data)
    cov  = code.get_covariance(data)
    
    titles = [
        ['a- original', 'b- kmeans applied with 25 clusters', 'c- slic applied with 934 clusters'],
        ['d- segmented a', 'e- segmented b', 'f- segmented c']
    ]
    
    for j in range(3):
        imgs[j] = code.skin_segmentation(images[j], mean, cov)
        
        axes[0][j].axis("off")
        axes[0][j].imshow(images[j])
        axes[0][j].set_title(titles[0][j])
        
        axes[1][j].axis("off")
        axes[1][j].imshow(imgs[j] ,cmap = 'gray')
        axes[1][j].set_title(titles[1][j])
        
    plt.savefig('results/demo3.png', transparent = Transparency)
    fig.tight_layout()
    return imgs


#########################################################
#                       DEMO 4                          #
#########################################################
def demo4(imgs1, imgs2):
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (8*2,8))
    
    titles = [
        ['a- original', 'b- kmeans applied with 25 clusters', 'c- slic applied with 934 clusters'],
        ['d- segmented a', 'e- segmented b', 'f- segmented c']
    ]
    
    for j in range(3):
        img, borders = code.postprocessing(imgs2[j]) 
        
        axes[0][j].axis("off")
        axes[0][j].imshow(imgs1[j])
        axes[0][j].set_title(titles[0][j])
        for b in borders: axes[0][j].add_patch(b)
        
        axes[1][j].axis("off")
        axes[1][j].imshow( img ,cmap = 'gray')
        axes[1][j].set_title(titles[1][j])
        
    
    plt.savefig('results/demo4.png', transparent = Transparency)
    fig.tight_layout()
    

#########################################################
#                       DEMO 5                          #
#########################################################
def demo5(batch, data):
    fig, axes = plt.subplots(nrows = 6, ncols = 3, figsize = (16,8*3))
    
    ids = 3*batch+1
    
    mean = code.get_mean(data)
    cov  = code.get_covariance(data)
    
    imgs = [None, None, None]
         
    for i in range(3):
        imgs[0] = code.read_image(ids+i, 'test')
        imgs[1],_ = code.kmeans(imgs[0], 30)
        imgs[2],_ = code.slic(imgs[0], 1000)
                
        for j in range(3):
            res = code.skin_segmentation(imgs[j], mean, cov)
            res,_ = code.postprocessing(res) 
            
            axes[i*2][j].axis("off")
            axes[i*2][j].imshow(imgs[j], 'gray')
            
            axes[i*2+1][j].axis("off")
            axes[i*2+1][j].imshow(res, 'gray')
    
    plt.savefig('results/demo5.'+str(batch+1)+'.png', transparent = Transparency)
    fig.tight_layout()
    plt.close()

    
#########################################################
#                      RUN DEMO                         #
#########################################################

def run_demos():
    data =code.sfa_dataset('skin')
    demo1(data)
    oimgs = demo2()
    rimgs = demo3(oimgs, data)
    demo4(oimgs, rimgs)
    
    demo5(0,data)
    demo5(1,data)
    demo5(2,data)
    
    
run_demos()
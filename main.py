import skin_segmentation as code #import the scipt
import numpy as np
import matplotlib.pyplot as plt

"""
Example Usage
"""

#read an image
img = plt.imread('data/test/1.jpg')

#specify the initial segmenting routine and number of segments
res = code.skin_segmentation_final(img, code.slic, 1000) 

#showing the image
plt.figure(figsize = (7,7))
plt.imshow(img)
plt.savefig('results/example.jpg')
plt.close()

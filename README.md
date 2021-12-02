# skin segmentation

The objective of the skin segmentation is to separate regions of skin and non-skin in colour images. This is done by classifying each pixel in the image as either being skin or not skin. This problem can be applied to a wide range of problems in computer vision, robotics and human-computer interaction. Examples of this are face detection and filtering crude images on the internet. Further details on the subject and implementation can be found on the report write up.

## Implemented techinque.
- The model that is implemented, fits a gaussuan distribution using a data set of skin pixels. Given an image, the model gives the likelyhood of a pixel being a skin pixel or not.
- The input images are pre-processed using clustering so as to reduce the amount of variation in the images.
- We use Otsu thresholding to produce the binary images.

## Results.
- Left images: no preprocessing
- middle images: k means clustering preprocessing
- right images: slic 

### Portrait images
![Alt text](https://https://github.com/kat-lego/skin_segmentation/blob/master/results/demo5.1.png)

### Fullbody images
![Alt text](https://https://github.com/kat-lego/skin_segmentation/blob/master/results/demo5.2.png)

### Group images
![Alt text](https://https://github.com/kat-lego/skin_segmentation/blob/master/results/demo5.3.png)

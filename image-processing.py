# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import lfilter

import os
# import skimage
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from skimage.util import crop

# load image
path = os.path.join('IMG_4346.jpeg')
img = io.imread(path)

# rotate image
img = rotate(img,-.25)

# grayscale image
gray_img = rgb2gray(img)


endx = 4032-3750
endy = 3024-1400

# crop image (beforex,afterx), (before y, aftery)
img_crop = crop(gray_img, ((1300,endy), (200,endx)))


# pull rows and columns for indexing
rows = img_crop[:,0]
cols = img_crop[0,:]

# zero array for binary input
shape=img_crop.shape
data=np.zeros(shape)
scatter=np.zeros(len(cols))


for i in range(len(cols)):          # loop through cols (horizontal)
    for n in range(len(rows)):      # loop through rows (vertical)
        if img_crop[n,i] < 0.425:   # if lighter than .____ grayscale
            data[n,i]=1             # value =1
        else:
            data[n,i]=0             # else value=0


for i in range(len(cols)):          # loop through cols (horizontal)
    ones = np.zeros(len(rows))      # create zero array to hold row indices, then resets every column iteration
    for n in range(len(rows)):      # loop through rows (vertical)
        if data[n,i]==1:
            ones[n]=n               # get index of every cell with value 1
    ones=ones[ones!=0]              # removes zeros from ones array
    scatter[i]=np.mean(ones)        # avg indices to get single value (height) for each row

# filtering
# series=pd.Series(scatter)
# roll=series.rolling(10).mean()

# for i in range(len(scatter)):
#    if scatter[i]>(roll[i]+2) or scatter[i]<(roll[i]-2):
#        scatter[i]=roll[i]


# debugging

#save values to file
#np.savetxt("pixels.csv", img, delimiter=",")


#displays
#x values
x=np.arange(len(cols))

#plots
plt.subplot(1,2,1)
plt.imshow(img)

# plt.subplot(2,2,2)
# plt.imshow(img_crop,cmap='Greys')

# plt.subplot(2,2,3)
# plt.imshow(data,cmap='Greys')

plt.subplot(1,2,2)
plt.scatter(x,scatter)

#fifty=np.ones(len(x))*50
#plt.scatter(x,fifty)

plt.ylim(0,len(rows))
#plt.gca().invert_yaxis()

plt.show()



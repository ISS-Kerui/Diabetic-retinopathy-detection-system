import os 
import numpy as np 
import cv2 
from scipy import misc
if __name__ == '__main__': 
	dirname = '/Volumes/Echo/image/train_ds2_crop/0'
	for picname in os.listdir(dirname):
		if picname.split('.')[-1] == 'jpg':
			picname = os.path.join(dirname,picname)
			img = cv2.imread(picname,0)         
			img = np.array(img)         
			mean = np.mean(img)         
			img = img - mean         
			img = img*1.2 + mean*0.7 
			# img = img/255.   
			misc.imsave(picname, img)
		
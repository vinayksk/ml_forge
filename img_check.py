import imghdr
import cv2
import os
import glob

image_path = '/tensorflow/models/research/vksk_testbed/images'

for f in glob.glob(image_path + '/*.jpg'):
    image = cv2.imread(f)
    file_type = imghdr.what(f)
    if file_type != 'jpeg':
        print(str(f) + "- invalid " + str(file_type))

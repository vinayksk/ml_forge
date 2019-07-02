import cv2
import numpy as np


for i in range(100):
    img = np.ones((480,640,3), np.uint8)
    img = img * np.random.randint(255)
    cv2.circle(img,(np.random.randint(300,340), np.random.randint(220, 260)), 220, (255,0,0), -1)
    img_name = "images/test_img_" + str(i) + ".jpg"
    cv2.imwrite(img_name, img)
 # Draw a diagonal blue line with thickness of 5 px



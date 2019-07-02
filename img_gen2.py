import cv2
import numpy as np
import csv
import os

num_img = 200
width = 640         # Image width
height = 480        # Image height 
cr_min = 2          # Min blob radius
cr_max = 10         # Max blob radius
max_num_blobs = 8   # Max number of blobs that will appear in the image
rbuffer = 2         # Extra border space around the enemy/ally blob
categories = {'enemy': {'color': (116, 116, 255), 'id': 1}, 'ally': {'color': (255, 123, 123), 'id': 2}}

with open('annotations.csv', mode='w') as csv_file:
    fieldnames = ['name', 'xmins', 'xmaxs', 'ymins', 'ymaxs', 'classes_text', 'classes']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(num_img):
        xmins, xmaxs, ymins, ymaxs, classes_text, classes = ([] for i in range(6))
        img = np.ones((height, width, 3), np.uint8)
        img = img * np.random.randint(255)
        for label in categories:
            for n in range(np.random.randint(1, max_num_blobs)):
                x = np.random.randint(cr_max + rbuffer, width - (cr_max + rbuffer + 1))
                y = np.random.randint(cr_max + rbuffer, height - (cr_max + rbuffer + 1))
                circle_radius = np.random.randint(cr_min, cr_max + 1)
                cv2.circle(img, (x, y), circle_radius, categories[label]['color'], -1)
                buffered_radius = circle_radius + rbuffer
                xmins.append(x - buffered_radius)
                xmaxs.append(x + buffered_radius)
                ymins.append(y - buffered_radius)
                ymaxs.append(y + buffered_radius)
                classes_text.append(label)
                classes.append(categories[label]['id'])
        img_name = "images/test_img_" + str(i) + ".jpg"
        cv2.imwrite(img_name, img)
        location = '/tensorflow/models/research/vksk_testbed/' + img_name
        writer.writerow({'name': location, 'xmins': xmins, 'xmaxs': xmaxs, 'ymins': ymins, 'ymaxs': ymaxs, 
            'classes_text': classes_text, 'classes': classes})



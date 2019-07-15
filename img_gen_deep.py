import cv2
import numpy as np
import csv
import os

class img_generator:
    def __init__(self, num_img = 3, cr_min = 2, cr_max = 2, max_num_blobs = 8, rbuffer = 2):
        self.num_img = num_img                # Number of images
        self.width = 640                      # Image width
        self.height = 480                     # Image height 
        self.cr_min = cr_min                  # Min blob radius
        self.cr_max = cr_max                  # Max blob radius
        self.max_num_blobs = max_num_blobs    # Max number of blobs that will appear in the image
        self.rbuffer = rbuffer                # Extra border space around the enemy/ally blob
        # pc = pixel count
        self.enemy_pc, self.ally_pc, self.bg_pc = (0 for i in range(3))
        self.categories = {'enemy': {'color': (116, 116, 255), 'id': 1}, 'ally': {'color': (255, 123, 123), 'id': 2}}

    def generate(self):
        with open('images/annotations.csv', mode='w') as csv_file:
            fieldnames = ['name', 'xmins', 'xmaxs', 'ymins', 'ymaxs', 'classes_text', 'classes']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(self.num_img):
                xmins, xmaxs, ymins, ymaxs, classes_text, classes = ([] for i in range(6))
                img = np.ones((self.height, self.width, 3), np.uint8)
                img = img * np.random.randint(255)
                for label in self.categories:
                    for n in range(np.random.randint(1, self.max_num_blobs)):
                        x = np.random.randint(self.cr_max + self.rbuffer, self.width - (self.cr_max + self.rbuffer + 1))
                        y = np.random.randint(self.cr_max + self.rbuffer, self.height - (self.cr_max + self.rbuffer + 1))
                        circle_radius = np.random.randint(self.cr_min, self.cr_max + 1)
                        cv2.circle(img, (x, y), circle_radius, self.categories[label]['color'], -1)
                        buffered_radius = circle_radius + self.rbuffer
                        xmins.append(x - buffered_radius)
                        xmaxs.append(x + buffered_radius)
                        ymins.append(y - buffered_radius)
                        ymaxs.append(y + buffered_radius)
                        classes_text.append(label)
                        classes.append(self.categories[label]['id'])
                img_name = "images/test_img_" + str(i) + ".jpg"
                cv2.imwrite(img_name, img)
                location = '/tensorflow/models/research/vksk_testbed/' + img_name
                writer.writerow({'name': location, 'xmins': xmins, 'xmaxs': xmaxs, 'ymins': ymins, 'ymaxs': ymaxs, 
                    'classes_text': classes_text, 'classes': classes})
            
                seg_class = np.ones((self.height, self.width, 1), np.uint8)
                for h in range(self.height):
                    for w in range(self.width):
                        if img[h, w][0] == 116 and img[h, w][1] == 116 and img[h, w][2] == 255:
                            self.enemy_pc = self.enemy_pc + 1
                            seg_class[h , w] = 1
                        elif img[h, w][0] == 255 and img[h, w][1] == 123 and img[h, w][2] == 123:
                            self.ally_pc = self.ally_pc + 1
                            seg_class[h, w] = 2
                        else:
                            self.bg_pc = self.bg_pc + 1
                            seg_class[h, w] = 0
                seg_name = "images/seg_img_" + str(i) + ".png"
                cv2.imwrite(seg_name, seg_class)
                        
        max_pc = max(self.enemy_pc/self.num_img, self.ally_pc/self.num_img, self.bg_pc/self.num_img)

        with open('images/averages.csv', mode='w') as av_file:
            fieldnames = ['enemy', 'ally', 'bg']
            av_writer = csv.DictWriter(av_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
            av_writer.writeheader()
            av_writer.writerow({'enemy': max_pc/(self.enemy_pc/self.num_img), 'ally': max_pc/(self.ally_pc/self.num_img), 'bg': max_pc/(self.bg_pc/self.num_img)})

gen1 = img_generator()
gen1.generate()

import os
from shutil import copyfile

file = set()
img_path = '/Users/prasadgujar16/Desktop/searching/my-images/a.jpeg'
dest = '/Users/prasadgujar16/Desktop/searching/'
copyfile(img_path, dest + '.jpeg')



import os

inputs = "/Users/prasadgujar16/Desktop/searching/my-images/c.jpg"
output = "/Users/prasadgujar16/Desktop/searching/my-images/a.jpg"

if os.path.exists(output):
    os.remove(output)

try:
    os.rename(inputs,output)
except OSError:
    pass
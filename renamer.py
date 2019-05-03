import os
import shutil
rename = ""
with open('caption.txt','r') as data:
    for line in data:
        rename +=line
        rename +=" "
    data.close()
rename = rename[:-1]
k1 = "/Users/prasadgujar16/Desktop/searching/my-images/"
cname = k1 + rename + ".jpg"
output = "/Users/prasadgujar16/Desktop/searching/my-images/"
shutil.copy(cname,output)

print(cname)
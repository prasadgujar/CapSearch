import os
from difflib import SequenceMatcher
from shutil import copyfile
import shutil
import glob
from PIL import Image
results = []
testdir = "/Users/prasadgujar16/Desktop/searching/ok/img"

b = ""
with open('query.txt','r') as data:
		for line in data:
			b +=line
files = glob.glob('/Users/prasadgujar16/Desktop/searching/static/people_photo/*')
for f in files:
    os.remove(f)
for f in os.listdir(testdir):
    pname = testdir
    if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'): 
        a = f
        #print (a)
        #b = "a group"
        #print (b)
        ratio = SequenceMatcher(None, a, b).ratio()
        #print(ratio)
        if ratio >= 0.2:
            path = (os.path.abspath(f))
            try:  
                img  = Image.open(path)  
            except IOError: 
                pass
            #print(os.path.splitext(os.path.basename(f))[0])
            name = os.path.splitext(os.path.basename(f))[0]
            #print (name)
            fname = pname +"/"+ name+".jpg"
            file = set()
            #img_path = '/Users/prasadgujar16/Desktop/searching/my-images/'
            #src = img_path + fname
            dest = '/Users/prasadgujar16/Desktop/searching/static//people_photo/'
            shutil.copy(fname, dest)
            print (fname)
            print(" ")

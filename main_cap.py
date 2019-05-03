import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #surpress tensorflow warnings
from capgen import CaptionGenerator
'''try:
	from query import description_search
except:
	print('Please ensure the elastic search server is enabled')
    '''
from PIL import Image
c = CaptionGenerator()
#import index
#import query
PATH_TO_FLICK8K_IMG="imgs/"
img_path = 'my-images/a.jpg'
caption = c.get_caption(img_path)
#os.remove('caption.txt')
wr = open('caption.txt', 'w')
wr.write(caption)
wr.close()

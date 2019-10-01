import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import sys
import os

argv = sys.argv
if len(argv) != 5:
	print("Invalid arguments")
	exit()
im = Image.open(argv[1])
width = 200 if argv[2] is None else int(argv[4])
x = 0 if argv[2] is None else int(argv[2])
y = 0 if argv[3] is None else int(argv[3])
area = [x, y, x+width, y+width]

im_crop = im.crop(area)
draw = ImageDraw.Draw(im)
draw.rectangle(area, outline='yellow', width=10)

#im_crop.show()
im.show()
def concat(x):
	y = ''
	for x_ in x:
		y += x_+'/'
	return y

if not os.path.exists('cropped_images/'+concat(argv[1].split('/')[:-1])):
	os.makedirs('cropped_images/'+concat(argv[1].split('/')[:-1]))

im.save('cropped_images/'+argv[1].split('.')[0]+'_box.'+argv[1].split('.')[-1])
im_crop.save('cropped_images/'+argv[1].split('.')[0]+'_cropped.'+argv[1].split('.')[-1])
print('saved', 'cropped_images/'+argv[1].split('.')[0]+'_cropped.'+argv[1].split('.')[-1])

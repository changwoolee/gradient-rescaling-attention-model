import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

lv_dir = './image_predictions_lv_99'
attention_dir = './image_predictions_lva'
block_dir = './image_predictions_lvab_99'

dset = 'DIV2K_valid'
filename='0845x4'

fig, axes = plt.subplots(1, 4, figsize=(30, 15))
for ax in axes:
	ax.set_axis_off()

ax1, ax2, ax3, ax4 = axes

#image = Image.open(os.path.join('/home/esoc/datasets/SuperResolution',dset, filename+'.png'))
image = Image.open(os.path.join('/home/esoc/datasets/SuperResolution/DIV2K_valid_HR', '0845.png'))
lv = np.load(os.path.join(lv_dir, dset+'_LR_bicubic/X4',filename+'.npy'))
lva = np.load(os.path.join(attention_dir, dset+'_LR_bicubic/X4',filename+'.npy'))
lvab = np.load(os.path.join(block_dir, dset+'_LR_bicubic/X4',filename+'.npy'))

lv = (lv).mean(axis=2)
lva =(lva).mean(axis=2)
lvab = (lvab).mean(axis=2)

minval = np.minimum(np.minimum(lva.min(), lvab.min()), lv.min())
maxval = np.maximum(np.maximum(lva.max(), lvab.max()), lv.max())

im1 = ax1.imshow(image)
im2 = ax2.imshow(lv, cmap='viridis', vmin=minval, vmax=maxval)
im3 = ax3.imshow(lva, cmap='viridis', vmin=minval, vmax=maxval)
im4 = ax4.imshow(lvab, cmap='viridis', vmin=minval, vmax=maxval)
#cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im4, ax=axes.ravel().tolist(), shrink=0.3)
#cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=10)
plt.axis('off')
#plt.rcParams.update({'font.size':50})

#plt.show()
plt.savefig('figure_compare.pdf')
plt.imsave('compare/im1.png', image)
plt.imsave('compare/im2.png', lv, cmap='viridis', vmin=minval, vmax=maxval)
plt.imsave('compare/im3.png', lva,cmap='viridis', vmin=minval, vmax=maxval)
plt.imsave('compare/im4.png', lvab,cmap='viridis', vmin=minval, vmax=maxval)

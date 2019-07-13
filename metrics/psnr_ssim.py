import numpy as np
import PIL.Image as Image
from skimage.measure import compare_psnr, compare_ssim
import glob
import os
import sys
import utils
import tqdm

csv_filename = sys.argv[1]
DATA_DIR='/home/esoc/datasets/SuperResolution/'
RESULT_DIR=sys.argv[2]

Benchmarks=['Set5', 'Set14', 'DIV2K_valid_HR', 'BSDS100', 'Urban100']
#Benchmarks=['Urban100']

def rgb_to_Y(img):
	xform = np.array(
			[[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0],
			[- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
			[112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
	img = img.dot(xform.T)[:,:,0:1]+16.0
	return img



for bn in Benchmarks:
	print(bn)
	data_dir = os.path.join(DATA_DIR, bn)
	result_dir = os.path.join(RESULT_DIR, bn)
	hr_images = sorted(glob.glob(data_dir+'/*.png'))
	sr_images = sorted(glob.glob(result_dir+'/x4/*_sr.png'))

	psnr_mean = []
	psnr_bic_mean = []
	ssim_mean = []
	ssim_bic_mean = []
	for hr_fp, sr_fp in zip(hr_images, sr_images):
		print(hr_fp, sr_fp)
		hr = Image.open(hr_fp).convert('RGB')
		sr = Image.open(sr_fp).convert('RGB')
		hr = hr.crop((0,0,sr.size[0],sr.size[1]))
		bicubic = hr.resize((hr.size[0]//4, hr.size[1]//4), Image.BICUBIC).resize((hr.size[0]//4*4, hr.size[1]//4*4), Image.BICUBIC)

#		hr = hr.convert('YCbCr')
#		sr = sr.convert('YCbCr')
#		bicubic = bicubic.convert('YCbCr')
		
		bicubic = rgb_to_Y(np.array(bicubic).astype(np.float64))
		hr_arr = rgb_to_Y(np.array(hr).astype(np.float64))
		sr_arr = rgb_to_Y(np.array(sr).astype(np.float64))


		cutoff = 6 + 4
		hr_arr = hr_arr[cutoff:-cutoff,cutoff:-cutoff,:]
		sr_arr = sr_arr[cutoff:-cutoff,cutoff:-cutoff,:]
		bicubic = bicubic[cutoff:-cutoff,cutoff:-cutoff,:]
		psnr_val = compare_psnr(hr_arr, sr_arr, data_range=255)
		psnr_bic_val = compare_psnr(hr_arr, bicubic, data_range=255)
		print(psnr_val)

		ssim_val = compare_ssim(hr_arr, sr_arr, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03, sigma=1.5, data_range=255)
		ssim_bic_val = compare_ssim(hr_arr, bicubic, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03, sigma=1.5, data_range=255)
		print(ssim_val)
		psnr_mean.append(psnr_val)
		psnr_bic_mean.append(psnr_bic_val)
		ssim_mean.append(ssim_val)
		ssim_bic_mean.append(ssim_bic_val)
	pm = np.array(psnr_mean).mean()
	pbm = np.array(psnr_bic_mean).mean()
	sm = np.array(ssim_mean).mean()
	sbm = np.array(ssim_bic_mean).mean()
	print('psnr:',pm,'psnr_bicubic:',pbm,'ssim:',sm, 'ssim_bicubic:',sbm)

	res = {'psnr_bicubic':pbm, 'psnr_pred':pm, 'ssim_bicubic':sbm, 'ssim_pred':sm}
	utils.save_csv(csv_filename, res, result_dir, data_dir)

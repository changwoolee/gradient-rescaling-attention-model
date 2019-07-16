import os
import glob
import logging
import argparse
import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from data import fullsize_sequence, DOWNGRADES
from model import load_model
from util import reset_session

DATASETS = ['Set5_LR_bicubic', 'Set14_LR_bicubic', 'DIV2K_valid_LR_bicubic', 'BSDS100_LR_bicubic', 'Urban100_LR_bicubic']

logger = logging.getLogger(__name__)


def model_paths(input_dir):
	path_pattern = os.path.join(input_dir, '**', '*.h5')
	paths = glob.glob(path_pattern, recursive=True)
	paths.sort()
	return paths


def predict_model(model_path, image_path, scale, outdir):
	logger.info('Load model %s', model_path)
	model = load_model(model_path, compile=False)

	for dataset_name in DATASETS:
		print(dataset_name)
		target_outdir = os.path.join(outdir, dataset_name, 'X%d'%(scale))
		target_image_path = os.path.join(image_path, dataset_name, 'X%d'%(scale))
		if not os.path.exists(target_outdir):
			os.makedirs(target_outdir)

		image_filenames = sorted(glob.glob(os.path.join(target_image_path, '*.png')))
		if len(image_filenames) == 0:
			logger.warning('No image files. Stop prediction..')
			exit()

		logger.info('Start prediction with %s', model_path)
		for i, f in tqdm.tqdm(enumerate(image_filenames)):
			filename = os.path.split(f)[1]
			im = Image.open(f).convert('RGB')
			im = np.array(im)
			im = np.reshape(im, (1,) + im.shape)
			output = model.predict(im, batch_size=1)
			output = np.squeeze(output)
			im_out = output
			if output.shape[-1] > 3:
				im_out = output[:,:,:3]
				logvar = output[:,:,3:]
				logvar_filename = os.path.join(target_outdir, os.path.splitext(filename)[0])
				np.save(logvar_filename, logvar)
				plt.imsave(logvar_filename+'_logvar.png', logvar.mean(axis=-1), cmap='viridis')
				
			im_out = Image.fromarray(np.uint8(im_out))
			im_out.save(os.path.join(target_outdir, filename))



def main(args):
	"""
	Evaluate all models in a user-defined directory against the DIV2K validation set.

	The results are written to a user-defined JSON file. All models in the input
	directory must have been trained for the same downgrade operator (bicubic or
	unknown) and the same scale (2, 3 or 4).
	"""

	mps = model_paths(args.indir)

	if mps:
		reset_session(args.gpu_memory_fraction)
		predict_model(mps[-1], args.dataset, args.scale, args.outdir)

	else:
		logger.warning('No models found in %s', args.indir)


def parser():
	parser = argparse.ArgumentParser(description='Evaluation against DIV2K validation set')

	parser.add_argument('-d', '--dataset', type=str, default='/home/esoc/datasets/SuperResolution/',
						help='path to DIV2K dataset with images stored as numpy arrays')
	parser.add_argument('-i', '--indir', type=str,
						help='path to models directory')
	parser.add_argument('-o', '--outdir', type=str, default='./image_predictions',
						help='output JSON file')
	parser.add_argument('-s', '--scale', type=int, default=2, choices=[2, 3, 4],
						help='super-resolution scale')
	parser.add_argument('--downgrade', type=str, default='bicubic', choices=DOWNGRADES,
						help='downgrade operation')
	parser.add_argument('--gpu-memory-fraction', type=float, default=0.8,
						help='fraction of GPU memory to allocate')

	return parser

if __name__ == '__main__':
	logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

	main(parser().parse_args())

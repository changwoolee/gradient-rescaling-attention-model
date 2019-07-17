import os
import argparse
import logging
import numpy as np
import tensorflow as tf

from callback import learning_rate
from data import DIV2KSequence, DOWNGRADES
from model import srgan, edsr
from train import create_train_workspace, write_args
from util import concurrent_generator, init_session

from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.applications.vgg19 import preprocess_input

from keras import backend as K

logger = logging.getLogger(__name__)


def content_loss_wrapper(pred_logvar=False, attention=False, block_attention_gradient=False):
	def content_loss(hr, sr):
		vgg = srgan.vgg_54()
		if pred_logvar:
			logvar = sr[:,:,:,3:]
			if attention:
				attention_mask = K.sigmoid(logvar)
			else:
				attention_mask = 1.
			if block_attention_gradient:
				attention_mask = K.stop_gradient(attention_mask)
		sr = sr[:,:,:,:3]
		sr_vgg = preprocess_input(sr)
		hr_vgg = preprocess_input(hr)
		sr_features = vgg(sr_vgg)
		hr_features = vgg(hr_vgg)
		loss = mean_squared_error(hr_features, sr_features)
		if pred_logvar:
			print(sr.shape)
			unc_loss = K.square(hr - sr)/(127.5**2)
			unc_loss = 127.5**2 * K.exp(-logvar)*unc_loss + logvar
				
			unc_loss = attention_mask * unc_loss
			unc_loss = K.mean(loss, axis=[1,2], keepdims=True)
			loss = unc_loss
	
		return loss
	return content_loss


def main(args):
	train_dir, models_dir = create_train_workspace(args.outdir)
	losses_file = os.path.join(train_dir, 'losses.csv')
	write_args(train_dir, args)
	logger.info('Training workspace is %s', train_dir)

	sequence = DIV2KSequence(args.dataset,
							 scale=args.scale,
							 subset='train',
							 downgrade=args.downgrade,
							 image_ids=range(1,801),
							 batch_size=args.batch_size,
							 crop_size=96)

	if args.generator == 'edsr-gen':
		generator = edsr.edsr_generator(args.scale, args.num_filters, args.num_res_blocks)
	else:
		generator = srgan.generator(args.num_filters, args.num_res_blocks, pred_logvar=args.pred_logvar)

	if args.pretrained_model:
		generator.load_weights(args.pretrained_model, by_name=True)

	for l in generator.layers:
		if 'conv2d_38' not in l.name:
			l.trainable=False
	generator.summary()

	generator_optimizer = Adam(lr=args.generator_learning_rate)

	discriminator = srgan.discriminator()
	discriminator_optimizer = Adam(lr=args.discriminator_learning_rate)
	discriminator.compile(loss='binary_crossentropy',
						  optimizer=discriminator_optimizer,
						  metrics=[])

	gan = srgan.srgan(generator, discriminator)
	gan.compile(loss=[content_loss_wrapper(args.pred_logvar, args.attention, args.block_attention_gradient), 'binary_crossentropy'],
				loss_weights=[0.006, 0.001] if not args.pred_logvar else [1., 0.],
				optimizer=generator_optimizer,
				metrics=[])

	generator_lr_scheduler = learning_rate(step_size=args.learning_rate_step_size, decay=args.learning_rate_decay, verbose=0)
	generator_lr_scheduler.set_model(gan)

	discriminator_lr_scheduler = learning_rate(step_size=args.learning_rate_step_size, decay=args.learning_rate_decay, verbose=0)
	discriminator_lr_scheduler.set_model(discriminator)

	with open(losses_file, 'w') as f:
		f.write('Epoch,Discriminator loss,Generator loss\n')

	with concurrent_generator(sequence, num_workers=1) as gen:
		for epoch in range(args.epochs):

			generator_lr_scheduler.on_epoch_begin(epoch)
			discriminator_lr_scheduler.on_epoch_begin(epoch)

			d_losses = []
			g_losses_0 = []
			g_losses_1 = []
			g_losses_2 = []

			for iteration in range(args.iterations_per_epoch):

				# ----------------------
				#  Train Discriminator
				# ----------------------

				lr, hr = next(gen)
				sr = generator.predict(lr)
				if args.pred_logvar:
					sr = sr[:,:,:,:3]

				hr_labels = np.ones(args.batch_size) + args.label_noise * np.random.random(args.batch_size)
				sr_labels = np.zeros(args.batch_size) + args.label_noise * np.random.random(args.batch_size)

				hr_loss = 0. #discriminator.train_on_batch(hr, hr_labels)
				sr_loss = 0. # discriminator.train_on_batch(sr, sr_labels)

				d_losses.append((hr_loss + sr_loss) / 2)

				# ------------------
				#  Train Generator
				# ------------------

				lr, hr = next(gen)

				labels = np.ones(args.batch_size)

				perceptual_loss = gan.train_on_batch(lr, [hr, labels])

				g_losses_0.append(perceptual_loss[0])
				g_losses_1.append(perceptual_loss[1])
				g_losses_2.append(perceptual_loss[2])

				print(f'[{epoch:03d}-{iteration:03d}] '
					  f'discriminator loss = {np.mean(d_losses[-50:]):.3f} '
					  f'generator loss = {np.mean(g_losses_0[-50:]):.3f} ('
					  f'mse = {np.mean(g_losses_1[-50:]):.3f} '
					  f'bxe = {np.mean(g_losses_2[-50:]):.3f})')

			generator_lr_scheduler.on_epoch_end(epoch)
			discriminator_lr_scheduler.on_epoch_end(epoch)

			with open(losses_file, 'a') as f:
				f.write(f'{epoch},{np.mean(d_losses)},{np.mean(g_losses_0)}\n')

			model_path = os.path.join(models_dir, f'generator-epoch-{epoch:03d}.h5')
			print('Saving model', model_path)
			generator.save(model_path)


def parser():
	parser = argparse.ArgumentParser(description='GAN training with custom generator')

	parser.add_argument('-o', '--outdir', type=str, default='./output',
						help='output directory')


	# --------------
	#  Uncertainty
	# --------------


	parser.add_argument('-plv', '--pred_logvar', action='store_true',
						help='If true predict log variance to estimate'
							'heteroscedastic uncertainty.')
	parser.add_argument('-att', '--attention', action='store_true',
						help='If true use attention.')
	parser.add_argument('-bag', '--block_attention_gradient', action='store_true',
						help='If true block attention gradient.')

	# --------------
	#  Dataset
	# --------------

	parser.add_argument('-d', '--dataset', type=str, default='./DIV2K_BIN',
						help='path to DIV2K dataset with images stored as numpy arrays')
	parser.add_argument('-s', '--scale', type=int, default=4, choices=[4],
						help='super-resolution scale')
	parser.add_argument('--downgrade', type=str, default='bicubic', choices=DOWNGRADES,
						help='downgrade operation')

	# --------------
	#  Model
	# --------------

	parser.add_argument('-g', '--generator', type=str, default='edsr-gen', choices=['edsr-gen', 'sr-resnet'],
						help='generator model name')
	parser.add_argument('--num-filters', type=int, default=64,
						help='number of filters in generator')
	parser.add_argument('--num-res-blocks', type=int, default=16,
						help='number of residual blocks in generator')
	parser.add_argument('--pretrained-model', type=str,
						help='path to pre-trained generator model')

	# --------------
	#  Training
	# --------------

	parser.add_argument('--epochs', type=int, default=150,
						help='number of epochs to train')
	parser.add_argument('--iterations-per-epoch', type=int, default=1000,
						help='number of update iterations per epoch')
	parser.add_argument('--batch-size', type=int, default=16,
						help='mini-batch size for training')
	parser.add_argument('--generator-learning-rate', type=float, default=1e-4,
						help='generator learning rate')
	parser.add_argument('--discriminator-learning-rate', type=float, default=1e-4,
						help='discriminator learning rate')
	parser.add_argument('--learning-rate-step-size', type=int, default=100,
						help='learning rate step size in epochs')
	parser.add_argument('--learning-rate-decay', type=float, default=0.1,
						help='learning rate decay at each step')
	parser.add_argument('--label-noise', type=float, default=0.05,
						help='amount of noise added to labels for discriminator training')

	# --------------
	#  Hardware
	# --------------

	parser.add_argument('--gpu-memory-fraction', type=float, default=0.8,
						help='fraction of GPU memory to allocate')

	return parser


if __name__ == '__main__':
	logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

	args = parser().parse_args()
	init_session(args.gpu_memory_fraction)
	main(args)

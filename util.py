import tensorflow as tf

from contextlib import contextmanager
from PIL import Image

from keras import backend as K
from keras.utils.data_utils import OrderedEnqueuer

def heteroscedastic_loss(attention=False, 
												 block_attention_gradient=False, 
												 mode='l2'):
	''' Heteroscedastic loss.'''

	def het_loss(y_true, y_pred):
		y_mean = y_pred[:,:,:,:3]
		y_logvar = y_pred[:,:,:,3:]
		if mode == 'l2':
			euclidian_loss = K.square(y_true - y_mean)/(127.5**2)
		elif mode == 'l1':
			euclidian_loss = K.abs(y_true/127.5 - y_mean/127.5)

		loss = 127.5 * tf.exp(-y_logvar)*euclidian_loss + y_logvar
		if mode == 'l2':
			loss *= 127.5
			

		if attention:
			attention_mask = tf.nn.sigmoid(y_logvar)

			if block_attention_gradient:
				attention_mask = tf.stop_gradient(attention_mask)

			loss = attention_mask * loss
		return K.mean(loss, axis=-1)

	return het_loss








@contextmanager
def concurrent_generator(sequence, num_workers=8, max_queue_size=32, use_multiprocessing=False):
	enqueuer = OrderedEnqueuer(sequence, use_multiprocessing=use_multiprocessing)
	try:
		enqueuer.start(workers=num_workers, max_queue_size=max_queue_size)
		yield enqueuer.get()
	finally:
		enqueuer.stop()


def init_session(gpu_memory_fraction):
	K.tensorflow_backend.set_session(tensorflow_session(gpu_memory_fraction=gpu_memory_fraction))


def reset_session(gpu_memory_fraction):
	K.clear_session()
	init_session(gpu_memory_fraction)


def tensorflow_session(gpu_memory_fraction):
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
	return tf.Session(config=config)


def load_image(path):
	img = Image.open(path)
	if img.mode != 'RGB':
		img = img.convert('RGB')
	return img

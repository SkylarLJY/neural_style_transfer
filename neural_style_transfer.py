import tensorflow as tf 
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model 
import numpy as np 
from style_content_model import StyleContentModel
import cv2
import PIL.Image
import IPython.display as display

style_weight = 1e-2
content_weight = 1e4

def tensor_to_image(tensor):
	tensor = tensor*255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor)>3:
		assert tensor.shape[0]==1
		tensor = tensor[0]
	return PIL.Image.fromarray(tensor)

def load_image(image):
	max_dim = 512
	img = cv2.imread(image)
	img = tf.image.convert_image_dtype(img, tf.float32)
	shape = tf.cast(tf.shape(img)[:-1], tf.float32)
	long_dim = max(shape)
	scale = max_dim/long_dim
	new_shape = tf.cast(shape*scale, tf.int32)
	img = tf.image.resize(img, new_shape)
	img = img[tf.newaxis, :]
	return img


def vgg_layers(layer_names):
	vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
	vgg.trainable = False
	outputs = [vgg.get_layer(name).output for name in layer_names]
	model = tf.keras.Model([vgg.input], outputs)
	return model


def clip_0_1(image):
 	return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


if __name__ == '__main__':
	content = load_image('content.jpg')
	style = load_image('style.jpeg')
	# vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
	
	content_layers = ['block5_conv2']
	style_layers = ['block1_conv1', 
					'block2_conv1', 
					'block3_conv1',
					'block4_conv1',
					'block5_conv1']
	num_content_layers = len(content_layers)
	num_style_layers = len(style_layers)

	extractor = StyleContentModel(style_layers, content_layers)
	results = extractor(tf.constant(content))
	style_targets = extractor(style)['style']
	content_targets = extractor(content)['content']

	image = tf.Variable(content)
	opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

	def total_loss(outputs):
		style_outputs = outputs['style']
		content_outputs = outputs['content']
		style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
				for name in style_outputs.keys()])
		style_loss *= style_weight/len(style_layers)

		content_loss = tf.add_n ([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
				for name in content_outputs.keys()])

		loss = style_loss + content_loss
		return loss

	def train_step(image):
		with tf.GradientTape() as tape:
			outputs = extractor(image)
			loss = total_loss(outputs)
		grad = tape.gradient(loss, image)
		opt.apply_gradients([(grad, image)])
		image.assign(clip_0_1(image))

	# train_step(image)
	# train_step(image)
	# train_step(image)
	
	# tensor_to_image(image).show()


	
	
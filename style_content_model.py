import tensorflow as tf


def vgg_layers(layer_names):
	vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
	vgg.trainable = False
	outputs = [vgg.get_layer(name).output for name in layer_names]
	model = tf.keras.Model([vgg.input], outputs)
	return model


def gram_matrix(tensor):
	result = tf.linalg.einsum('bijc, bijd->bcd', tensor, tensor)
	input_shape = tf.shape(tensor)
	num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
	return result/num_locations
	# temp = tensor
	# # squeeze -> remove all dementions with size 1
	# temp = tf.squeeze(temp)
	# fun = tf.reshape(temp, [temp.shape[2], temp.shape[0]*temp.shape[1]])
	# result = tf.matmul(temp, temp, transpose_b=True)
	# gram = tf.expand_dims(result, axis=0)
	# return gram 


class StyleContentModel(tf.keras.models.Model):
	def __init__(self, style_layers, content_layers):
		super(StyleContentModel, self).__init__()
		self.vgg = vgg_layers(style_layers+content_layers)
		self.style_layers = style_layers
		self.content_layers = content_layers
		self.num_style_layers = len(style_layers)
		self.vgg.trainable = False

	def __call__(self, inputs):
		inputs = inputs*255.0
		preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
		outputs = self.vgg(preprocessed)
		style_outputs = outputs[:self.num_style_layers]
		content_outputs = outputs[self.num_style_layers:]

		style_outputs = [gram_matrix(s_out) for s_out in style_outputs]

		content_dict = {name:val for name, val in zip(self.content_layers, content_outputs)}
		style_dict = {name:val for name, val in zip(self.style_layers, style_outputs)}

		return {'content': content_dict, 'style': style_dict}

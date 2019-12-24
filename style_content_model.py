import tensorflow as tf


def mini_model(layer_names, model):
	outputs = [model.get_layer(name).output for name in layer_names]
	model = Model([vgg.input], output)
	return model

	
class StyleContentModel(tf.keras.models.Model):
	def __init__(self, style_layers, content_layers, vgg):
		super(StyleContentModel, self).__init__()
		self.vgg = mini_model(style_layers+content_layers, vgg)
		self.style_layers = style_layers
		self.content_layers = content_layers
		self.num_style_layers = len(style_layers)
		self.vgg.trainable = False

	def call(self, inputs):
		inputs = inputs*255.0
		preprocessed == preprocess_input(inputs)
		outputs = self.vgg(preprocessed)
		style_outputs = outputs[:self.num_style_layers]
		content_outputs = outputs[self.num_style_layers:]

		style_outputs = [gram_matrix(s_out) for s_out in style_outputs]

		content_dict = {name:val for name, val in zip(self.content_layers, content_outputs)}
		style_dict = {name:val for name val in zip(self.style_layers, style_outputs)}

		return {'content': content_dict, 'style': style_dict}

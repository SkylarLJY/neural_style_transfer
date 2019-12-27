import time
import sys
import tensorflow as tf
import IPython.display as display
from neural_style_transfer import train_step, tensor_to_image, load_image
from style_content_model import StyleContentModel

epochs = 10
step_per_epoch = 100
total_variation_weight = 30

if __name__ == '__main__':
	content = load_image('content.jpg')
	style = load_image('style.jpeg')
	
	content_layers = ['block5_conv2']
	style_layers = ['block1_conv1', 
					'block2_conv1', 
					'block3_conv1',
					'block4_conv1',
					'block5_conv1']

	extractor = StyleContentModel(style_layers, content_layers)
	results = extractor(tf.constant(content))
	style_targets = extractor(style)['style']
	content_targets = extractor(content)['content']

	image = tf.Variable(content)
	opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

	start = time.time()
	step = 0
	for n in range(epochs):
		for m in range (step_per_epoch):
			step += 1
			train_step(image, extractor, opt, style_targets, content_targets)
			print('.', end='')
			sys.stdout.flush()
		display.clear_output(wait=True)
		print()
	tensor_to_image(image).save("result.png")

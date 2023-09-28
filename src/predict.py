import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import logging


logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
logging.warning('logger started')

# Load the trained model
model = load_model('model/cifar10_model.h5')

# Load an image for prediction (you can replace this with your own image)
# Make sure the image size matches the input shape of the model
# image = ...  # Load your image here

# Preprocess the image (resize, normalize, etc.) to match model input
# Replace this with your own preprocessing logic

# Make a prediction

# making a custom function to preprocess the image and to run the inferene on the image
def predict_image(image_path):
	labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()

	# pdb.set_trace()
	# load the image in an array

	logging.info(f'Loading and preprocessing {image_path} image')
	img = load_img(image_path, target_size=(32, 32))
	
	logging.info(f'converting image to array')
	image_n = img_to_array(img)

	# reshape it
	p = image_n.reshape(1, 32, 32, 3)
	 
	# prepare pixel data
	p = p.astype('float32')
	p= p / 255.0
	
	# pass in the network for prediction and
	# save the predicted label
	logging.info(f'model inference in process')
	logging.info(f'::::::::::::::::::::::::::::::')
	logging.info(f'::::::::::::::::::::::::::::::')
	logging.info(f'::::::::::::::::::::::::::::::')
	logging.info(f'::::::::::::::::::::::::::::::')
	
	predicted_label = labels[model.predict(p).argmax()]
	logging.info(f'::::::::::::::::::::::::::::::')
	logging.info(f'::::::::::::::::::::::::::::::')
	logging.info(f'::::::::::::::::::::::::::::::')
	logging.info(f'::::::::::::::::::::::::::::::')
	
	logging.info(f'the input file is {image_path}')
	

	
	return predicted_label

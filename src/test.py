from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

def load_file(filename):
	# ladataan oma tesikuva
	image = load_img(filename, grayscale=True, target_size=(28, 28))
	image = img_to_array(image)
	# reshapetaan kuva
	image = image.reshape(1, 28, 28, 1)
	# normalisoidaan kuva
	image = image.astype('float32')
	image = image / 255.0
	return image

def run_test():
	# määritetään mikä kuva ja mikä malli
	image = load_file('test_image_0.png')
	# käytetään tallennettua mallia
	model = load_model('DigitRecognition_model.h5')
	# Arvataan kuvassa oleva numero
	prediction = model.predict(image)
	digit = argmax(prediction)
	# printataan numero
	print(digit)
    
run_test()
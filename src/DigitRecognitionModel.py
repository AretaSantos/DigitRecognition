from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

# luvun tunnistus minst datan avulla
# Treenidata sisältää 60 000 esimerkkiä 28x28 koossa
# Testidata sisältää 10 000

def load_data():
    # ladataan data minst.load_data() avulla
    # jaetaan data treeni ja testidataan
    # X sisältää datan, Y on label
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # Neuroverkot vaativat datan kolmiulotteisena datana
    # reshapetaan ladatut data arrayt, jotta ne saavat "single color channelin"
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encodataan to_categorical() funktiolla labelit numeeriseen muotoon,
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    # palautetaan reshapeattu ja hot encodattu data
    return trainX, trainY, testX, testY

    # normalisoidaan harmaanväsyisten kuvien pikseleiden arvot olemaan 0-1 välillä
def pixel_prep(train, test):
    # muutetaan integerit float muotoisiksi
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalisoidaan
    train_norm = train_norm/255.0
    test_norm = test_norm/255.0
    return train_norm, test_norm


def define_model():
	model = Sequential()
    # luodaan convolutional layer, kokona 3x3, filtterimääränä 32.  Määritellän Input, joka on 28x28. 
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    # maxpooling ottaa isoimman arvon alkuperäisestä matriisista
	model.add(MaxPooling2D((2, 2)))
    # flätätään yksiulotteiseksi vektoriksi
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # softmax antaa todenäköisyydet oppimisen perusteella
    # viimeisellä layerillä yhtä paljon kuin numeroluokkia (0-9 = 10)
    # softmax skaalaa tulokset 0-1 välillä
	model.add(Dense(10, activation='softmax'))
	# compilataan malli, 
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

  
    # arvioidaan malli käyttämällä k-fold Cross-validationia
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# luodaan cross validation, jaetaan viiteen osaan, sekoitetaan data ennen jakamista
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		#määritellään malli
		model = define_model()
		#valitaan treeni ja testidata
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model(), treenaa mallia datalla annetulla epochs(10) kertaa validoidaan testidatalla
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		#arvioidaan malli, printataan accuracy
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		#tallenetaan score ja histories kuvantamista varten
		scores.append(acc)
		histories.append(history)
	return scores, histories

#piirretään kuvaaja
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()

def summarize_performance(scores):
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))

#ajetaan ohjelma ja kutsutaan kaikkia funkitoita
def run_test():
   trainX, trainY, testX, testY = load_data()
   #trainX, testX = pixel_prep(trainX, testX)
   scores, histories = evaluate_model(trainX, trainY)
   summarize_diagnostics(histories)
   summarize_performance(scores)

run_test()


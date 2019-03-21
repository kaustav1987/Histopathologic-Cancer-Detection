import base64
import numpy as np
import io
import os
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from keras.models import model_from_json
##from keras.layers.core import Dense, Flatten
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, BatchNormalization,MaxPooling2D,Dense,Dropout,Flatten,Activation


app = Flask(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_model():
    global model
    #vgg_model = VGG16(include_top = False, input_shape = (96,96,3), weights ='imagenet')

    #model = Sequential()
    #model.add(vgg_model)
    #model.add(Flatten())
    #model.add(Dense(1024, activation = 'relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(512, activation = 'relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1, activation = 'sigmoid'))
    global graph
    graph = tf.get_default_graph()

    ##load json and create model
    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    ##json_file.close()
    #model = model_from_json(loaded_model_json)
    ##model.load_weights("model_weights_0303.h5")
    model = load_model('model_march03_latest.h5')
    ##model = load_model('VGG16_cats_and_dogs.h5')
    ##model = load_model('model.h5')
    print(" * Model loaded!")
    ##model.save("model.h5")
    ##print('New Model saved')

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    print('Kaustav \n')
    ##image = Image.open(io.BytesIO(decoded))
    dataBytesIO = io.BytesIO(decoded)
    dataBytesIO.seek(0)
    image =  Image.open(dataBytesIO)
    processed_image = preprocess_image(image, target_size=(96, 96))
    with graph.as_default():
        prediction = model.predict(processed_image).tolist()

    print(prediction)
    print(type(prediction))
    #not_cancer= (1 - prediction).astype(float)
    #cancer= prediction.astype(float)

    #print(type(cancer))
    #print(type(not_cancer))

    response = {
        'prediction': {
            'Cancer': prediction[0][0],
            ##'Not Cancer': prediction[0][0]
        }
    }
    return jsonify(response)



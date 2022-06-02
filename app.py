from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.preprocessing import image


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)


MODEL_PATH = 'models/encoder_dense.h5'


model = load_model(MODEL_PATH)
# model._make_predict_function()         
print('Model loaded. Start serving...')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64,64))

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255
   
    preds = model.predict(img)
    pred = np.argmax(preds,axis = 1)
    return pred


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        
        pred = model_predict(file_path, model)
        os.remove(file_path) 


        
        str1 = 'The given blood sample is parasitized/infected. Please contact the medical authorities immediately.'
        str2 = 'The given blood sample is normal/uninfected. Stay safe!'
        if pred[0] == 1:
            return str1
        else:
            return str2
    return None

    
# if __name__ == '__main__':
#         app.run()


http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()

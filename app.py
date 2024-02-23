import os
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
import tensorflow as tf
from flask import Flask, url_for, render_template, redirect, request
import sqlite3 as SQL
app = Flask(__name__)

# from keras.applications.vgg16 import preprocess_input,VGG16
global graph
UPLOAD_FOLDER = './static/uploader/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SIZE = 120


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/<string:page_name>')
def html_page(page_name):
    return render_template(page_name)


@app.route('/upload', methods=['POST', 'GET'])
def Upload():
    if request.method == 'POST':
        print(request)
        file = request.files['image']
        print('file', file)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], '1.png'))

        model = keras.models.load_model(r'./model/model_16.h5')
        categories = ["NonDemented", "MildDemented",
                      "ModerateDemented", "VeryMildDemented"]

        nimage = cv2.imread(r"./static/uploader/1.png", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(nimage, (SIZE, SIZE))
        image = image/255.0
        prediction = model.predict(np.array(image).reshape(-1, SIZE, SIZE, 1))
        pclass = np.argmax(prediction)
        pValue = "Predict: {0}".format(categories[int(pclass)])
        print(pValue)
        realvalue = "Real Value 1"
        print('success')
        img = "./static/uploader/1.png"
        return render_template('result.html', value=pValue)

    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)

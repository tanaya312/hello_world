# Import necessary libraries
from flask import Flask, render_template, request
from flaskwebgui import FlaskUI  # get the FlaskUI class

import numpy as np
import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load model
model = load_model("model/v4_pred_cott_dis.h5")

print('@@ Model loaded')


def pred_cot_dieas(cott_plant):
    test_image = load_img(cott_plant, target_size=(150, 150))  # load image
    print("@@ Got Image for prediction")

    # convert image to np array and normalize
    test_image = img_to_array(test_image)/255
    # change dimention 3D to 4D
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image).round(
        3)  # predict diseased plant or not
    print('@@ Raw result = ', result)

    pred = np.argmax(result)  # get the index of max value

    if pred == 0:
        return "Diseased Leaf", 'healthy_plant_leaf.html'  # if index 0 burned leaf
    elif pred == 1:
        return "Healthy Plant", 'healthy_plant.html'   # if index 1
    elif pred == 3:
        return 'Diseased Plant', 'disease_plant.html'
    else:
        return "Healthy Plant", 'healthy_plant.html'  # if index 3

# ------------>>pred_cot_dieas<<--end


# Create flask instance
app = Flask(__name__)
# Feed it the flask app instance
ui = FlaskUI(app)

# render index.html page

# do your logic as usual in Flask


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fet input
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_cot_dieas(cott_plant=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)


# call the 'run' method
ui.run()

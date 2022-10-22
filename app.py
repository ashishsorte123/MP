from __future__ import division, print_function
import os
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_mobilenet.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0) 

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Glioma Tumor → Glioma is a type of tumor that occurs in the brain and spinal cord. Gliomas begin in the gluey supportive cells (glial cells) that surround nerve cells and help them function. Three types of glial cells can produce tumors."
    elif preds==1:
        preds = "Meningioma Tumor → A meningioma is a tumor that forms in your meninges, which are the layers of tissue that cover your brain and spinal cord. They're usually not cancerous (benign), but can sometimes be cancerous (malignant). Meningiomas are treatable."
    elif preds == 2:
        preds = "No Tumor → Be Happy, There is was No Sign of Any Tumor in your MRI"
    else:
        preds = "Pituitary Tumor → A pituitary tumor is a tumor that forms in the pituitary gland near the brain that can cause changes in hormone levels in the body. This illustration shows a smaller tumor (microadenoma). Pituitary tumors are abnormal growths that develop in your pituitary gland."
    return preds

# (training = g --> 1 to 3, m --> 2 to 3, n --> 1 to 10, p --> 5 to 10)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        print(result)
        return result
    return None


if __name__ == '__main__':
    # app.run(debug=True)
    app.run()
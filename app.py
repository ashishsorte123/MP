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
# MODEL_PATH ='model_mobilenet.h5'
# MODEL_PATH ='model_resnet50.h5'
# MODEL_PATH ='model_vgg19.h5'
MODEL_PATH = 'model_mobilenet(realtime images).h5'
# MODEL_PATH = 'model_resnet50(real time images).h5'
# MODEL_PATH = 'model_vgg19(real time images).h5'
# MODEL_PATH = 'model_svm(realtime images).h5'
# MODEL_PATH = 'model_xception(realtime images).h5'
# MODEL_PATH = 'model_inceptionv3(realtime images).h5'
# MODEL_PATH = 'model_resnetv2(realtime images).h5'

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
        preds = "Glioma Tumor → A glioma tumor is a type of brain tumor that arises from the glial cells in the brain. Gliomas can be benign (non-cancerous) or malignant (cancerous) and can occur anywhere in the brain or spinal cord. Symptoms includes headaches, seizures, changes in vision or hearing., etc."
    elif preds==1:
        preds = "Meningioma Tumor → A meningioma is a tumor that forms in your meninges, which are the layers of tissue that cover your brain and spinal cord. They're usually not cancerous (benign), but can sometimes be cancerous (malignant). Meningiomas are treatable."
    elif preds == 2:
        preds = "No Tumor → Be Happy, There is was No Sign of Any Tumor in your MRI"
    else:
        preds = "Pituitary Tumor → A pituitary tumor is a tumor that forms in the pituitary gland near the brain that can cause changes in hormone levels in the body. This illustration shows a smaller tumor (microadenoma). Pituitary tumors are abnormal growths that develop in your pituitary gland."
    return preds

# (training = g --> 1 to 3, m --> 2 to 3, n --> 1 to 10, p --> 5 to 10)
# (training = g --> 530, m --> m3 242, n --> 162, p --> 782 )
@app.route('/', methods=['GET'])

@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/mobilenet1')
def mobilenet1():
    return render_template('mobilenet1.html')

@app.route('/mobilenet2')
def mobilenet2():
    return render_template('mobilenet2.html')

@app.route('/resnet1')
def resnet1():
    return render_template('resnet1.html')

@app.route('/resnet2')
def resnet2():
    return render_template('resnet2.html')

@app.route('/svm1')
def svm1():
    return render_template('svm1.html')

@app.route('/svm2')
def svm2():
    return render_template('svm2.html')

@app.route('/vgg1')
def vgg1():
    return render_template('vgg1.html')

@app.route('/vgg2')
def vgg2():
    return render_template('vgg2.html')

@app.route('/inception1')
def inception1():
    return render_template('inception1.html')

@app.route('/inception2')
def inception2():
    return render_template('inception2.html')

@app.route('/xception1')
def xception1():
    return render_template('xception1.html')

@app.route('/xception2')
def xception2():
    return render_template('xception2.html')

@app.route('/resv21')
def resv21():
    return render_template('resv21.html')

@app.route('/resv22')
def resv22():
    return render_template('resv22.html')

@app.route('/index')
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
    app.run(debug=True)
    # app.run()
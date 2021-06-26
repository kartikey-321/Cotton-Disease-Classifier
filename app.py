from flask import Flask,render_template,redirect,request
import numpy as np
from tensorflow import keras
import os
from werkzeug.utils import secure_filename
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

app=Flask(__name__)

model=keras.models.load_model('cotton_disease.h5')
# model._make_predict_function()


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="It is diseased cotton leaf"
    elif preds==1:
        preds="It is diseased cotton plant"
    elif preds==2:
        preds="It is fresh cotton leaf"
    else:
        preds="It is fresh cotton plant"
        
    
    
    return preds



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    if request.method=='POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        result=preds
    return render_template('index.html',data=result)












if __name__=='__main__':
    app.run(debug=True)
from flask import Flask,render_template,Response,request,make_response
import json,base64
import tensorflow as tf 
import cv2
from PIL import Image
import numpy as np
app=Flask(__name__)
model = tf.keras.models.load_model('content/age_pred')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
@app.route('/',methods=['GET', 'POST'])
def index():

    if  request.method == "POST":
        f=request.files['sentFile']
        response = {}
        image = Image.open(f)
        #image = image.resize((64,64),Image.ANTIALIAS)
        image =image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = np.array(image)
        if len(image.shape) > 2 and image.shape[2] == 4:
            #convert the image from RGBA2RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        pred=[['Face Not Detected']]
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = cv2.resize(image[y:y + h, x:x + w], dsize=(64, 64))
            data = np.asarray(face)
            batch = data.reshape(-1,64,64,3)
            pred = int(model.predict(batch)[0][0])
            cv2.putText(image, str(pred), (x+5, y+3), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2)
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        retval, buffer  = cv2.imencode('.png', image)
        #response['prediction']=str(pred)

        response['image']=base64.encodebytes(buffer).decode('utf-8')
        return Response(json.dumps( response ))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
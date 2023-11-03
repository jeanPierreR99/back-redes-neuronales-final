from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from tensorflow import keras
import os
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)
graph_directory = 'generator/'
class_tags = ['normal','fuego']
model = keras.models.load_model('modelo_fuego_hsv.h5')

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        images_prueba = []
        image = request.files['image'].read()
        npimg = np.fromstring(image, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
       
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.resize(hsv_image, (150, 150))
        images_prueba.append(hsv_image)

        original_image_path = os.path.join(graph_directory, 'original_image.jpg')
        processed_image_path = os.path.join(graph_directory, 'processed_image.jpg')

        cv2.imwrite(original_image_path, img)
        cv2.imwrite(processed_image_path, images_prueba[0])

        prediction = model.predict(np.array([hsv_image]))
        print("prediccion: ",prediction[0])
        print("argumentos ",np.argmax(prediction))
        print("la prediccion dice: ",class_tags[np.argmax(prediction)])
        
        if (class_tags[np.argmax(prediction)] == 'fuego'):
            result = "Fuego"
        else:
            result = "Normal"

        imagen_path1 = './generator/original_image.jpg'
        imagen_path2 = './generator/processed_image.jpg'  # Reemplaza con la ruta correcta de tu imagen

    # Lee la imagen y conviértela en una cadena base64
        with open(imagen_path1, 'rb') as image_file:
            image_base64_1 = base64.b64encode(image_file.read()).decode()

        with open(imagen_path2, 'rb') as image_file:
            image_base64_2 = base64.b64encode(image_file.read()).decode()


        return jsonify({'prediction': result, 'original_image':image_base64_1, 'processed_image':image_base64_2})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

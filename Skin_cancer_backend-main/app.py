import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---- Custom KANLayer Definition ----
class KANLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Names are required: the HDF5 format keys weights by name within a
        # layer, so two unnamed add_weight() calls collide on save/load.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.nn.silu(tf.matmul(inputs, self.kernel) + self.bias)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

# ---- Load Your Model ----
# Legacy HDF5, not .keras. best_model.keras was saved on Windows by Keras 2.x,
# which built the archive's internal HDF5 paths with os.path.join() -> backslash
# separators. HDF5 treats "\" as an ordinary character rather than a group
# separator, so on Linux the loader looks for "layers/stem_conv/vars/0", finds
# nothing, and fails with "expected 1 variables, but received 0". This .h5 was
# re-saved from that file on Windows and verified to give identical predictions.
MODEL_PATH = "best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'KANLayer': KANLayer})

# ---- Image Preprocessing ----
IMG_SIZE = 224  # Use your model's input size

def remove_hair(img):
    """Remove hair using morphological operations and inpainting."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def preprocess_image(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = remove_hair(img)
    return img.astype(np.float32)

# ---- Flask App Setup ----
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    try:
        img = preprocess_image(file.read())
    except ValueError:
        return jsonify({'error': 'Uploaded file is not a readable image'}), 400
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    result = "malignant" if pred >= 0.231 else "benign"
    return jsonify({'probability': float(pred), 'result': result})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8080)  # Changed port to 8080

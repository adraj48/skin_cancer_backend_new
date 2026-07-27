import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---- Interpreter ----
# TFLite, not TensorFlow. The full TF runtime is ~400MB resident, which leaves
# too little of Render's 512MB free tier for an inference pass -- the container
# was OOM-killed mid-request every time. tflite-runtime is ~5MB.
#
# tflite-runtime ships Linux wheels only, so fall back to TensorFlow's bundled
# interpreter for local development on Windows/macOS. Same graph either way.
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # pragma: no cover - dev machines without tflite-runtime
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# Converted from best_model.h5 float32, no quantisation: verified to match the
# Keras model to 3.3e-07, so the 0.231 threshold below is still calibrated.
MODEL_PATH = "model.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
_input = interpreter.get_input_details()[0]
_output = interpreter.get_output_details()[0]

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
    """Decode to a raw 0-255 float32 RGB tensor.

    Deliberately no /255 or mean-subtraction here: the graph starts with
    Rescaling(1/255) and Normalization(ImageNet mean/std) layers, so scaling
    again would feed the network inputs it was never trained on.
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = remove_hair(img)
    return img.astype(np.float32)

def predict_one(img):
    """Run a single image through the interpreter."""
    interpreter.set_tensor(_input['index'], np.expand_dims(img, axis=0))
    interpreter.invoke()
    return float(interpreter.get_tensor(_output['index'])[0][0])

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
    pred = predict_one(img)
    result = "malignant" if pred >= 0.231 else "benign"
    return jsonify({'probability': pred, 'result': result})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8080)  # Changed port to 8080

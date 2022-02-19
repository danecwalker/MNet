import base64
from flask import Flask, request, jsonify, render_template
import api.torch_helpers
from api.torch_helpers import predict_y, convert_image

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  try:
    imageb64 = request.get_json()
    # print(imageb64)
    if imageb64 is None:
      return jsonify({"error": "No Image"})

    header, encoded = imageb64["image"].split(',', 1)
    img_bytes = base64.b64decode(encoded)
    tensor = convert_image(img_bytes)
    prediction = predict_y(tensor)
    print(prediction)
    data = {"output": prediction.item()}
    return jsonify(data)

  except Exception as e:
    print(e)
    return jsonify({"error": "Prediction Error"})

def run(model="1645269116"):
  api.torch_helpers.set_model_path(model)
  app.run(debug=True, host='0.0.0.0', port=3000)
import io
from time import time
import torch
import torchvision.transforms as transforms

from PIL import Image, ImageOps

from api.net import MNet

# neural net parameters
input_size = 784 # 28*28
hidden_size = 100
output_size = 10

model = MNet(input_size, hidden_size, output_size)

MODEL_PATH = f'./models/mnist_api_1645269116.pth'
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

def set_model_path(_model):
  # load neural model
  MODEL_PATH = f'./models/mnist_api_{_model}.pth'
  model.load_state_dict(torch.load(MODEL_PATH, map_location='gpu'))
  model.eval()

# image convert to tensor
def convert_image(image_bytes):
  transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
                                  transforms.Resize((28, 28)),
                                  transforms.ToTensor()])

  image = Image.open(io.BytesIO(image_bytes))
  image = ImageOps.invert(image)
  # image.save(f'./api/imgs/image_{int(time())}.jpg')
  return transform(image).unsqueeze(0)

def predict_y(image_tensor):
  images = image_tensor.reshape(-1, 28*28)
  outputs = model(images)
  _, predicted = torch.max(outputs.data, 1)
  return predicted
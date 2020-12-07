from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import numpy as np

import argparse
import torch
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_resnet.h5'
weight_path='weight/model_best.pth.tar'
if torch.cuda.is_available():
    gpu=0
else:
    gpu=None
# Load your trained model

def load_model(weight_path,gpu):
    model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=2);
            
    if gpu is None:
        checkpoint = torch.load(weight_path,map_location='cpu')
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(weight_path, map_location=loc);
    best_acc1 = checkpoint['best_acc1']

    if type(best_acc1)!=torch.Tensor:
        best_acc1=torch.tensor(best_acc1)
    if gpu is not None:
        # best_acc1 may be from a checkpoint from a different GPU
        best_acc1 = best_acc1.to(gpu)
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    model.load_state_dict(checkpoint['state_dict']);
    return model

model=load_model(weight_path,gpu)
print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(testimg_path, model,gpu):
    classes=['cats', 'dogs']
    ## Same image transformation as in the training time
    Test_transformer=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor()])
    img_pil=Image.open(testimg_path)
    img_t=Test_transformer(img_pil).float()
    img_t=img_t.reshape([1]+list(img_t.shape))

    model.eval();
    with torch.no_grad():
        if gpu is not None:
            img_t = img_t.cuda(gpu, non_blocking=True)

        # compute output
        output = model(img_t)

        predicted_class=classes[np.argmax(output.tolist(),axis=1)[0]]
        # print('Predicted Category: ',predicted_class)
        # font=ImageFont.truetype('utils/SansSerif.ttf', size=40)
        # ImageDraw.Draw(img_pil).text((0, 0),predicted_class,(255, 0, 0),font=font)
        # img_pil.show()
    return predicted_class


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
            basepath, 'static', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        predicted_class = model_predict(file_path, model,gpu)
        os.remove(file_path)
        # Process your result for human
        result = str(predicted_class)               # Convert to string
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)


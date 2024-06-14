import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello_world():
  return render_template('login.html')
@app.route("/results.html")
def results():
  return render_template('results.html')
@app.route("/home.html")
def home():
  return render_template('home.html')
@app.route("/login.html")
def login():
  return render_template('login.html')
@app.route("/contact.html")
def contact():
  return render_template('contact.html')
@app.route("/method.html")
def method():
  return render_template('method.html')
@app.route("/execute_python_function", methods=["POST"])
def execute_python_function():
    
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img, (600, 200))
    img1 = img1.reshape(1, -1) / 255
    p = model.predict(img1)
    dec= {0:'glioma',1:'Meningioma',2:'notumor',3:'pituary'}
    res=""
    tumor_type = dec[p[0]]
    if tumor_type=='notumor':
      return "The Image Doesn't Contain Brain Tumor"
    elif tumor_type=='glioma':
      res="Brain Tumor is detected,The predicted type of tumor is "
    elif tumor_type=='Meningioma':
      res="Brain Tumor is detected, The predicted type of tumor is "
    else:
      res="Brain Tumor is detected,The predicted type of tumor is "
    return res+tumor_type
   
if __name__ == '__main__':
  app.run(host="localhost", port=9392, debug=True)

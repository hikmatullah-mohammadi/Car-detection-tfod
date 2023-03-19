# Car Detection TFOD
Car detection in real time using Tensorflow Object Detection API and python-opencv (cv2)

## Project overview:
Object detection is a popular use case of computer vision, and can be used to solve many real world problems. One use of object detection is car detection in real time, which, in turn, can help traffic flow analysis. Also, car detection in an essencial part of autonomous automobiles which assists them to detect other vehicles and navigate accordingly. Hence, I decided to build/fine-tune a *Single Shot Detection (SSD)* model to detect  cars on the streets in real time.

## Project structure:
First, the model is trained on *google colab* using Tensorflow Object Detection (TFOD) API, and then it is exported to *Tensorflow saved model* format after being evaluated. Next, using the *python-opencv* library and Tensorflow, the model is used to detect cars in real time video footage.

## How to run
1. Install required libraries
  ```
  pip install tensorflow
  pip install python-opencv
  ```
2. Run the following command
  ```
  python index.py
  ```

## Screenshots
Screensot #1<br>
![Screensot #1](https://raw.githubusercontent.com/hikmatullah-mohammadi/car-detection-tfod/master/scr_shots/sc_shot-1.JPG)
Screensot #2<br>
![Screensot #2](https://raw.githubusercontent.com/hikmatullah-mohammadi/car-detection-tfod/master/scr_shots/sc_shot-2.JPG)

## Usefull Links:
- [Open the model's notebook on google colab](https://colab.research.google.com/drive/1jasbCSqQ4TmDyVAm6TUACzPAdPbvxdoG?usp=sharing)
- [Dataset used- Car detection mini-dataset](https://www.kaggle.com/datasets/hikmatullahmohammadi/car-detection-tfod)
- [Tensorflow Object Detection API's Github repo](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Tensorflow Object Detection API tutorial/docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/)
- [OpenCV docs](https://docs.opencv.org/4.x/)
- [The link to the video used for testing](https://www.youtube.com/watch?v=KBsqQez-O4w)

## Author profiles:
- [Hikmatullah Mohammadi- Kaggle profile](https://www.kaggle.com/hikmatullahmohammadi)
- [Hikmatullah Mohammadi- Github repo](https://github.com/hikmatullah-mohammadi)
- [Hikmatullah Mohammadi- LinkedIn profile](https://www.linkedin.com/in/hikmatullah-mohammadi-871550225/)
import numpy as np
import cv2
import tensorflow as tf

import utils

# Load saved model and build the detection function
detect_fn = tf.saved_model.load('./exported model/saved_model')
cap = cv2.VideoCapture('./video/cars-highway.mp4')

count_going = 0
count_coming = 0
while True:
    success, img = cap.read()
    pure_img = img.copy()
    if success == 0:
        break
    
    
    height, width, c = img.shape
    
    ############################################################
    pure_img[:200] = [0, 0, 0] # exclude noisy areas: areas outside street
    input_tensor = tf.convert_to_tensor(np.expand_dims(pure_img, 0), dtype=tf.uint8)

    # detect objects
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    dt_boxes = detections['detection_boxes']
    dt_classes = detections['detection_classes']
    dt_scores = detections['detection_scores']

    img = utils.draw_bbs_on_detection(img, dt_boxes, dt_classes, dt_scores, {1: 'Car'})
    ############################################################

    cv2.imshow('', img)
        
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
    
cv2.destroyAllWindows()
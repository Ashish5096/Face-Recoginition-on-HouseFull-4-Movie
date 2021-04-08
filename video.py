from tensorflow.keras.models import load_model
from imutils.face_utils import FaceAligner
import time
import math
import cv2
import numpy as np
import pickle
import imutils


protoPath="Models/face_detection_model/deploy.prototxt"
modelPath="Models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(protoPath,modelPath)
embedder = load_model("Models/FaceNet/facenet_keras.h5")

pickle_in=open("HousefullClassifier.pickle","rb")
clf=pickle.load(pickle_in)

pickle_in=open("HousefullLabel.pickle","rb")
le=pickle.load(pickle_in)


conf_threshold=float(0.7)
cap=cv2.VideoCapture("Housefull 4.mp4")

while(True):

    fps_start_time=time.time()
    _,frame=cap.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    
    frame_pixels = np.asarray(frame)
    frame_pixels = frame_pixels.astype('float32')
    mean         = np.asscalar(frame_pixels.mean())
        
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (mean, mean, mean))
    detector.setInput(blob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue
            face = cv2.resize(face, (160, 160))
            face_pixels = np.asarray(face)
            face_pixels = face_pixels.astype('float32')

            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean)/std
        
            samples = np.expand_dims(face_pixels, axis=0)
            embedding = embedder.predict(samples)
            embedding = embedding[0].reshape(1,-1)

            preds = clf.predict_proba(embedding)[0]
            j = np.argmax(preds)
            proba = preds[j]          
            
            name = le.classes_[j]
            text = "{}: {:.2f}%".format(name, proba*100)

            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            if(startY -10 >10):
                y=startY-10
            else:
                y=startY+10

            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    time_diff = time.time() - fps_start_time
            
    if(time_diff ==0):
        fps =0
    else:
        fps = 1/ time_diff
                
    fps_text = "FPS {:.2f}".format(fps)
    cv2.putText(frame,fps_text,(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1)

    cv2.imshow('Live',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
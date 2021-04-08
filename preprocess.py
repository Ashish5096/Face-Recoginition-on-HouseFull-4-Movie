from imutils.face_utils import FaceAligner
import numpy as np
import cv2
import imutils
import os
import math
import dlib


protoPath="Models/face_detection_model/deploy.prototxt"
modelPath="Models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
predictor= dlib.shape_predictor("Models/face_alignment_model/shape_predictor_68_face_landmarks.dat")

detector = cv2.dnn.readNetFromCaffe(protoPath,modelPath)
fa = FaceAligner(predictor, desiredFaceWidth=160)
conf_threshold=float(0.42)

size=0.7                #Train Test Split
count=0

DATADIR = "Dataset/Housefull_Dataset"
SAVEDIR = "Dataset/Preprocess_Dataset1"

labels=os.listdir(DATADIR)

for (i, name) in enumerate(labels):
    
    path=os.path.join(DATADIR,name)
    n=math.floor(size*len(os.listdir(path)))
    count=0

    for image in os.listdir(path):
        imagePath=os.path.join(path,image)
        img=cv2.imread(imagePath,1)
        img = imutils.resize(img, width=600)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        (h, w) = img.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0:
		
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > conf_threshold:
			
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = img[startY:endY, startX:endX]
                rect=dlib.rectangle(startX,startY,endX,endY)
                faceAligned = fa.align(img, gray, rect)
                
                count +=1    
                if(count <= n):
                    savePath=os.path.join(SAVEDIR,"Train")
                else:
                    savePath=os.path.join(SAVEDIR,"Test")

                image_name=name+"_"+str(count)+".jpg"
                savepath=os.path.join(savePath,image_name)
            
                try:
                    cv2.imwrite(savepath,faceAligned)
                except Exception as e:
                    pass
  
                
print("Preprocessing Done")

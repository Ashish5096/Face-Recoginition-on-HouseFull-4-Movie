from tensorflow.keras.models import load_model
import cv2
import os
import pickle
import numpy as np


model = load_model("Models/FaceNet/facenet_keras.h5")

DATADIR = "Dataset/Preprocess_Dataset1"
labels=os.listdir(DATADIR)

for (i, name) in enumerate(labels):

    knownEmbeddings = []
    knownNames = []
    total=0
    path=os.path.join(DATADIR,name)

    for image in os.listdir(path):
        imagePath=os.path.join(path,image)
        img=cv2.imread(imagePath,1)
        face_pixels = np.asarray(img)
        face_pixels = face_pixels.astype('float32')

        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean)/std
        
        samples = np.expand_dims(face_pixels, axis=0)
        embedding = model.predict(samples)
        knownEmbeddings.append(embedding[0])
        
        image_name=image.split('_')[0]
        knownNames.append(image_name)
        
        total += 1

    print("[INFO] serializing "+name+" {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}

    pickle_out=open("HousefullEmbed"+name+".pickle","wb")
    pickle.dump(data,pickle_out)
    pickle_out.close() 
             

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import numpy as np


le = LabelEncoder()
in_encoder = Normalizer(norm='l2')

pickle_in=open("HousefullEmbedTrain.pickle","rb")
data=pickle.load(pickle_in)

x_train = data["embeddings"]
x_train = in_encoder.transform(x_train)

y_train = np.array(data["names"])
le.fit(y_train)
y_train = le.transform(y_train)



pickle_in=open("HousefullEmbedTest.pickle","rb")
data=pickle.load(pickle_in)

x_test= data["embeddings"]
x_test = in_encoder.transform(x_test)

y_test = np.array(data["names"])
y_test = le.transform(y_test)


clf = SVC(C=1.0, kernel="linear",probability=True)
clf.fit(x_train, y_train)

confidence=clf.score(x_test,y_test)
print("confidence:- ",confidence)


pickle_out=open("HousefullClassifier.pickle","wb")
pickle.dump(clf,pickle_out)
pickle_out.close()

pickle_out=open("HousefullLabel.pickle","wb")
pickle.dump(le,pickle_out)
pickle_out.close()


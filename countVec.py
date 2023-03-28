
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from build_data import *

text_vect = CountVectorizer(max_df=0.2).fit(train_data + test_data)
word_vect = text_vect.transform(train_data)
test_vect = text_vect.transform(test_data)


def pre_countVec(pre):
    global predicted
    if pre == "SVC":
        text_clf = SVC(C=50).fit(word_vect, train_flag)
        predicted = text_clf.predict(test_vect)
    elif pre == "Perceptron":
        text_clf = Perceptron().fit(word_vect, train_flag)
        predicted = text_clf.predict(test_vect)
    elif pre == "RandomForest":
        text_clf = RandomForestClassifier(n_estimators=250).fit(word_vect, train_flag)
        predicted = text_clf.predict(test_vect)
    return predicted
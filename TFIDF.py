from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from build_data import *

text_vect = TfidfVectorizer().fit(train_data + test_data)  # 词汇表建立，词汇表中词项的idf值计算
word_vect = text_vect.transform(train_data)  # 向量化表示
test_vect = text_vect.transform(test_data)


def pre_tfidf(pre):
    global predicted
    if pre == "SVC":
        text_clf = SVC().fit(word_vect, train_flag)
        predicted = text_clf.predict(test_vect)
    elif pre == "Perceptron":
        text_clf = Perceptron().fit(word_vect, train_flag)
        predicted = text_clf.predict(test_vect)
    elif pre == "RandomForest":
        text_clf = RandomForestClassifier().fit(word_vect, train_flag)
        predicted = text_clf.predict(test_vect)
    return predicted




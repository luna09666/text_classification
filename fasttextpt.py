import jieba
import warnings
import fasttext
from jieba import analyse
warnings.filterwarnings('ignore')
jieba.setLogLevel(jieba.logging.INFO)
from build_data import *


with open("data/train_new.txt", "w", encoding='utf-8', errors='ignore') as f:
    for i in range(len(train_data)):
        keywords = jieba.analyse.textrank(train_data[i], topK=10, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'))
        f.writelines("__label__" + train_flag[i] + " " + " ".join(keywords) + '\n')
f.close()

with open("data/test_new.txt", "w", encoding='utf-8', errors='ignore') as f:
    for i in range(len(test_data)):
        keywords = jieba.analyse.textrank(test_data[i], topK=10, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'))
        f.writelines("__label__" + test_flag[i] + " " + " ".join(keywords) + '\n')
f.close()


model = fasttext.train_supervised("data/train_new.txt",
                                    lr = 0.2, 
                                    dim = 41,
                                    epoch = 100,
                                    word_ngrams = 2, 
                                    loss = 'softmax')



with open("data/test_new.txt", 'rb') as f:
    test = f.readlines()
f.close()
test_new = []
for i in test:
    test_new.append(" ".join(i.decode('utf-8').split()[1:]))

predicted = []
tag, acc = model.predict(test_new)
predicted = [i[0].replace("__label__", "") for i in tag]

# print(metrics.classification_report(test_flag, predicted))

# 数据预处理
from random import *
import jieba

with open("data/neg.txt", 'rb') as f:
    neg = f.readlines()
f.close()

with open("data/pos.txt", 'rb') as f:
    pos = f.readlines()
f.close()


print(len(pos), len(neg))

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def movestopwords(sentence):
    seq_list = jieba.lcut(sentence)
    stopwords = stopwordslist('data/stopwords.txt')
    santi_words = [x for x in seq_list if len(x) > 1 and x not in stopwords]
    return ''.join(santi_words)

train_data = []
train_flag = []
test_data = []
test_flag = []

for i in sample(neg, 1500):
    i = i.decode('utf-8')
    li = i.split()
    if(len(li) == 2):
        train_data.append(movestopwords(li[1]))
    else:
        train_data.append('')
    train_flag.append(li[0])

for i in sample(pos, 3500):
    i = i.decode('utf-8')
    li = i.split()
    if (len(li) == 2):
        train_data.append(movestopwords(li[1]))
    else:
        train_data.append('')
    train_flag.append(li[0])

for i in sample(neg, 1500):
    i = i.decode('utf-8')
    li = i.split()
    if (len(li) == 2):
        test_data.append(movestopwords(li[1]))
    else:
        test_data.append('')
    test_flag.append(li[0])

for i in sample(pos, 3500):
    i = i.decode('utf-8')
    li = i.split()
    if (len(li) == 2):
        test_data.append(movestopwords(li[1]))
    else:
        test_data.append('')
    test_flag.append(li[0])

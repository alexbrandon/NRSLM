#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
import gensim
import codecs
import ujson as json
#import path
import time
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='questionLstm.log',
                    filemode='w')

logging.debug('This is debug message')
logging.info('This is info message')
logging.warning('This is warning message')


def load_config():
    global modelPath
    print(os.getcwd())
    path = os.path.join("../"+"config/config")
    with open(path,"r") as file:
        line = file.read()
        line =str(line)
        line_json = json.loads(line)
    modelPath = line_json["modelPath"]
    logging.info("load config sucess!")


def load_wordvec(word):
    global model
    return model[word]


def load_w2c_model():
    global model
    global modelPath
    model = gensim.models.Word2Vec.load(modelPath)
    logging.info("load model sucess!")

'''def questionLstm():
    global model
    with open('ThePeddlerSpy.txt','r') as f:
        text = f.read()
    vocab = set(text)
    vocab_to_int = {c: i for i ,c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    encoded = np.array([vocab_to_int[c] for c in text],dtype=np.int32)'''


def init():
    global model
    global modelPath
    load_config()
    load_w2c_model()



if __name__ == '__main__':
    print(8*'*' + " Welcome " + 8*'*')
    init()
    word = "清华大学"
    w2c_vec = load_wordvec(word)
    print(w2c_vec)

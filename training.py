
import json
import nltk
import numpy
import pickle
import tflearn
import tensorflow as tf

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

class Training:
    words = []
    labels = []
    word_tokens = []
    intents = []
    training = []
    output = []

    def __init__(self, intent_file, cache):
        with open(intent_file) as file:
            self.data = json.load(file)
        
        self.cache = cache
        try:
            with open(self.cache, "rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)
        except:
            for intent in self.data["intents"]:
                for pattern in intent["patterns"]:
                    wrds = nltk.word_tokenize(pattern)
                    self.words.extend(wrds)
                    self.word_tokens.append(wrds)
                    self.intents.append(intent["tag"])
                
                if intent["tag"] not in self.labels:
                    self.labels.append(intent["tag"])

            
            self.words = [stemmer.stem(w.lower()) for w in self.words if w != "?"]
            self.words = sorted(list(set(self.words)))


    def generate_words_bag(self):
        out_empty = [0 for _ in range(len(self.labels))]

        for x, doc in enumerate(self.word_tokens):
            bag = []
            wrds = [stemmer.stem(w) for w in doc]

            for w in self.words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
        
            output_row = out_empty[:]
            output_row[self.labels.index(self.intents[x])] = 1

            self.training.append(bag)
            self.output.append(output_row)
        
        self.training = numpy.array(self.training)
        self.output = numpy.array(self.output)

        with open(self.cache, "wb") as f:
            pickle.dump((self.words, self.labels, self.training, self.output),f)
    

    def train_model(self):
        tf.compat.v1.reset_default_graph()

        self.net = tflearn.input_data(shape=[None, len(self.training[0])])
        self.net = tflearn.fully_connected(self.net, 8)
        self.net = tflearn.fully_connected(self.net, 8)
        self.net = tflearn.fully_connected(self.net, len(self.output[0]), activation="softmax")
        self.net = tflearn.regression(self.net)

        self.model = tflearn.DNN(self.net)
        
    
    def fit_model(self, output):
        try:
            self.model.load(output)
        except:
            self.train_model()
            self.model.fit(self.training, self.output, n_epoch=1000, batch_size=8, show_metric=True)
            self.model.save(output)
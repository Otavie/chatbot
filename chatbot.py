import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn as tfl
import tensorflow as tf
import random
import json
import pickle

with open("chatbot1Intents.json") as file:
    data = json.load(file)

wordList = []
labelList = []
wordTokens = []
intentList = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words = nltk.word_tokenize(pattern)
        wordList.extend(words)
        wordTokens.append(words)
        intentList.append(intent["tag"])

    if intent["tag"] not in labelList:
        labelList.append(intent["tag"])

wordList = [stemmer.stem(w.lower()) for w in wordList if w != "?"]
wordList = sorted(list(set(wordList)))

labelList = sorted(labelList)

#print(labelList)

training = []
output = []

outputEmpty = [0 for _ in range(len(labelList))]

for count, label in enumerate(wordTokens):
    bag = []

    words = [stemmer.stem(w.lower()) for w in label]

    for w in wordList:
        if w in words:
            bag.append(1)
        else:
            bag.append(0)

        
    outputRow = outputEmpty[:]
    outputRow[labelList.index(intentList[count])] = 1

    training.append(bag)
    output.append(outputRow)

training = np.array(training)
output = np.array(output)

tf.compat.v1.get_default_graph()

net = tfl.input_data(shape=[None, len(training[0])])
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, len(output[0]), activation="softmax")
net = tfl.regression(net)

model = tfl.DNN(net)

model.save("model.tflearn")
model.fit(training, output, n_epoch=1000, batch_size=10, show_metric=True)

def BagOfWords(s, wordList):
    bag = [0 for _ in range(len(wordList))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(wordList):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        userInput = input("You: ")
        if userInput.lower() == "quit":
            break

        results = model.predict([BagOfWords(userInput, wordList)])[0]
        #round(results, 5)
        resultsIndex = np.argmax(results)
        tag = labelList[resultsIndex]

        if results[resultsIndex] > 0.3:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I do not understand you. Say something else!")

        #print(round(results, 5))

        print(results)

chat()
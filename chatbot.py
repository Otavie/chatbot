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
wordTokenList = []
intentList = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words = nltk.word_tokenize(pattern)
        wordList.extend(words)
        wordTokenList.append(words)
        intentList.append(intent["tag"])

    if intent["tag"] not in labelList:
        labelList.append(intent["tag"])

wordList = [stemmer.stem(w.lower()) for w in wordList if w != "?"]
wordList = sorted(list(set(wordList)))

labelList = sorted(labelList)

#print(labelList)

trainingList = []
outputList = []

outputEmpty = [0 for _ in range(len(labelList))]

for count, wordToken in enumerate(wordTokenList):
    bagList = []

    words = [stemmer.stem(thisWord.lower()) for thisWord in wordToken]

    for thisWord in wordList:
        if thisWord in words:
            bagList.append(1)
        else:
            bagList.append(0)

        
    outputRow = outputEmpty[:]
    outputRow[labelList.index(intentList[count])] = 1

    trainingList.append(bagList)
    outputList.append(outputRow)

trainingList = np.array(trainingList)
outputList = np.array(outputList)

tf.compat.v1.get_default_graph()

net = tfl.input_data(shape=[None, len(trainingList[0])])
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, len(outputList[0]), activation="softmax")
net = tfl.regression(net)

model = tfl.DNN(net)

model.save("model.tflearn")
model.fit(trainingList, outputList, n_epoch=1000, batch_size=10, show_metric=True)


def BagOfWords(userSentence, wordList):
    bagList = [0 for _ in range(len(wordList))]

    sentTokenList = nltk.word_tokenize(userSentence)
    sentTokenList = [stemmer.stem(word.lower()) for word in sentTokenList]

    for sentToken in sentTokenList:
        for ind, w in enumerate(wordList):
            if w == sentToken:
                bagList[ind] = 1
            
    return np.array(bagList)

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
            for thisTag in data["intents"]:
                if thisTag['tag'] == tag:
                    responses = thisTag['responses']
            print(random.choice(responses))
        else:
            print("I do not understand you. Say something else!")

        #print(round(results, 5))

        print(results)

chat()
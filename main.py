import nltk
import random
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


from training import Training

t = Training("chatbot1Intents.json", "first.pickle")

if len(t.training) == 0 and len(t.output) == 0:
    t.generate_words_bag()

t.train_model()

t.fit_model("first.tflearn")

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

        results = t.model.predict([BagOfWords(userInput, t.words)])[0]
        #round(results, 5)
        resultsIndex = np.argmax(results)
        tag = t.labels[resultsIndex]

        if results[resultsIndex] > 0.3:
            for thisTag in t.data["intents"]:
                if thisTag['tag'] == tag:
                    responses = thisTag['responses']
            print(random.choice(responses))
        else:
            print("I do not understand you. Say something else!")

        #print(round(results, 5))

        print(results)

chat()
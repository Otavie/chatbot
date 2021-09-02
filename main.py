import json
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

t2 = Training("chatbot2Intents.json", "second.pickle")
if len(t2.training) == 0 and len(t2.output) == 0:
    t2.generate_words_bag()
t2.train_model()
t2.fit_model("second.tflearn")

t3 = Training("chatbot3Intents.json", "third.pickle")
if len(t3.training) == 0 and len(t3.output) == 0:
    t3.generate_words_bag()
t3.train_model()
t3.fit_model("third.tflearn")

t4 = Training("chatbot4Intents.json", "fourth.pickle")
if len(t4.training) == 0 and len(t4.output) == 0:
    t4.generate_words_bag()
t4.train_model()
t4.fit_model("fourth.tflearn")


def BagOfWords(userSentence, wordList):
    bagList = [0 for _ in range(len(wordList))]

    sentTokenList = nltk.word_tokenize(userSentence)
    sentTokenList = [stemmer.stem(word.lower()) for word in sentTokenList]

    for sentToken in sentTokenList:
        for ind, w in enumerate(wordList):
            if w == sentToken:
                bagList[ind] = 1
            
    return np.array(bagList)


def bot(userInput, t):
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
    
    return results


def greeting():
    print("Hi, I am Otavie Jnr, a bot created by my owner Otavie. How can I help you today?")
    print("Press 1 to know more about my creator's area of interest.")
    print("Press 2 to know about the awesome things he's worked on")
    print("Press 3 if you seek technical consultation in frontend related stuff")
    print("Press 4 if you'd just like to here one of my lame jokes.")


def chat():
    arr = ["1", "2", "3", "4"]
    greeting()
    while True:
        userInput = input("You: ")
        if userInput.lower() == "quit":
            print("Goodbye!")
            break
        
        if userInput == "1":
            print("To go back to the previous menu, press 0")
            print()
            while True:
                newInput = input("You: ")
                if newInput != "0":
                    print(bot(newInput, t))
                else:
                    greeting()
                    break
        
        if userInput == "2":
            print("To go back to the previous menu, press 0")
            print()
            while True:
                newInput = input("You: ")
                if newInput != "0":
                    print(bot(newInput, t2))
                else:
                    greeting()
                    break
        
        if userInput == "3":
            print("To go back to the previous menu, press 0")
            print()
            while True:
                newInput = input("You: ")
                if newInput != "0":
                    print(bot(newInput, t3))
                else:
                    greeting()
                    break
        
        if userInput == "4":
            print("To go back to the previous menu, press 0")
            print()
            while True:
                newInput = input("You: ")
                if newInput != "0":
                    print(bot(newInput, t4))
                else:
                    greeting()
                    break

        while userInput not in arr:
            print("You need to select a valid item to begin this conversation")
            print()
            break

chat()
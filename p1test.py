# CS114B Spring 2021 Programming Assignment 1
# Naive Bayes Classifier and Evaluation

from operator import truediv
import os
from re import I
from turtle import circle
import numpy as np
from collections import defaultdict

class NaiveBayes():

    def __init__(self):
        # be sure to use the right class_dict for each data set
        self.class_dict = {}
        # self.class_dict = {'neg': 0, 'pos': 1}
        #self.class_dict = {'action': 0, 'comedy': 1}
        self.feature_dict = {}
        self.prior = None
        self.likelihood = None

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    '''
    def train(self, train_set):
        self.feature_dict = self.select_features(train_set)
        # iterate over training documents
        nDoc = 0
        V = {}
        words = []
        nC = []
        if (len(self.class_dict) != 0):
            nC = [0] * len(self.class_dict)
            for key in self.class_dict:
                    V.update({key: {}})
        for root, dirs, files in os.walk(train_set):
            if (len(self.class_dict) == 0):
                i = 0
                for dir in dirs:
                    self.class_dict[dir] = i
                    i += 1
                for key in self.class_dict:
                    V.update({key: {}})
                nC = [0] * len(self.class_dict)
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # collect class counts and feature counts
                    # c = root.split('/')[2]
                    c = os.path.basename(root)
                    temp = f.read().split()
                    nDoc += 1
                    nC[self.class_dict[c]] += 1
                    for word in temp:
                        if word not in V[c]:
                            V[c][word] = 1
                            if word not in words:
                                words.append(word)
                        else:
                            V[c][word] += 1
                    
        # Make sure that the words that don't appear in a class are still represented in the Vocabulary
        totalWords = 0
        for c in V:
            for w in words:
                if w not in V[c]:
                    V[c][w] = 0
            
        self.prior = np.zeros((len(self.class_dict),))
        self.likelihood = np.zeros((len(self.class_dict), len(self.feature_dict)))
        for c in self.class_dict:
            self.prior[self.class_dict[c]] = np.log(nC[self.class_dict[c]]/nDoc)
            for f in self.feature_dict:
                totalWords = np.sum(list(V[c].values()))
                self.likelihood[self.class_dict[c]][self.feature_dict[f]] = np.log((V[c][f] + 1) / (totalWords + len(V[c])))
        print(self.prior)
        return self.prior, self.likelihood, V
        # normalize counts to probabilities, and take logs

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        featureVectors = {}
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # create feature vectors for each document
                    c = os.path.basename(root)
                    correctC = self.class_dict[c]
                    results[name]["correct"] = correctC
                    fV = np.zeros(len(self.feature_dict))
                    temp = f.read().split()
                    for word in temp:
                        if word in self.feature_dict.keys():
                            fV[self.feature_dict[word]] += 1
                # get most likely class
                likelihoodPerClass = self.likelihood @ fV
                sum = self.prior + likelihoodPerClass
                predictedC = np.argmax(sum)
                results[name]["predicted"] = predictedC

        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you may find this helpful
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        for doc in results.values():
            confusion_matrix[doc["predicted"]][doc["correct"]] += 1
        precisions = np.zeros(len(self.class_dict))
        recalls = np.zeros(len(self.class_dict))
        f1scores = np.zeros(len(self.class_dict))
        for c in self.class_dict:
            truePositive = confusion_matrix[self.class_dict[c]][self.class_dict[c]]
            retrieved = sum(confusion_matrix[:,self.class_dict[c]])
            relevant = sum(confusion_matrix[self.class_dict[c],])

            if (retrieved == 0): retrieved = 0.1
            precision = truePositive / retrieved
            if (relevant == 0): relevant = 0.1
            recall = truePositive / relevant
            f1score = 2 * precision * recall / ((precision + recall) if precision + recall > 0 else 0.1)
            
            precisions[self.class_dict[c]] = precision
            recalls[self.class_dict[c]] = recall
            f1scores[self.class_dict[c]] = f1score
        
        totalTruePositives = np.trace(confusion_matrix)
        totalEverything = np.sum(confusion_matrix)
        accuracy = totalTruePositives / totalEverything
        for k, v in self.class_dict.items():
            print(k, ":")
            print("Precision: ", precisions[v])
            print("Recall: ", recalls[v])
            print("F1: ", f1scores[v])
        print("Accuracy: ", accuracy)

    '''
    Performs feature selection.
    Returns a dictionary of features.
    '''
    def select_features(self, train_set):
        # almost any method of feature selection is fine here
        myDict = {}
        index = 0
        myArray = []
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    words = f.read().split()
                    for word in words:
                        if word not in myDict:
                            myDict[word] = index
                            myArray.append(1)
                            index += 1
                        else:
                            myArray[myDict[word]] += 1
        
        numFeatures = int(np.ceil(len(myArray) * .10)) # take top 10% of words
        features = {}

        for i in range (numFeatures):
            topInd = np.argmax(myArray)
            feat = [key for key, value in myDict.items() if topInd == value][0]
            features[feat] = i
            myArray[topInd] = 0
        
        return features
        # return {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}

if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')
    #nb.train('movie_reviews_small/train')
    results = nb.test('movie_reviews/dev')
    # print(results)
    #results = nb.test('movie_reviews_small/test')
    nb.evaluate(results)

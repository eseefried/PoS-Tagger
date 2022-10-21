# CS542 Fall 2021 Programming Assignment 2
# Logistic Regression Classifier

import os
from select import select
import numpy as np
from collections import defaultdict
from math import ceil
from random import Random
import pandas as pd

'''
Computes the logistic function.
'''
def sigma(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression():

    def __init__(self, n_features=20):
        # be sure to use the right class_dict for each data set
        self.class_dict = {'neg': 0, 'pos': 1}
        #self.class_dict = {'action': 0, 'comedy': 1}
        # use of self.feature_dict is optional for this assignment
        #self.feature_dict = {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}
        self.feature_dict = {'fun': 0, 'good': 1, 'great': 2, 'awesome': 3, 'funny':4, 'bad':5, 'terrible': 6, 'sucks' : 7, 
                              'laughed':8, 'cried': 9, 'boring': 10, 'brilliant': 11, 'dreadful': 12, 'difficult': 13,
                              'great': 14, 'embarrassment': 15, 'risk': 16, 'recommend': 17, 'stop': 18, 'waste': 19, 'dumb': 20}
                              #'wrong': 21, 'horible':22, "obvious": 23, "cheesy":24, "long":25, "fantastic": 26, "entertaining":27,
                              #"exciting": 28}
        self.n_features = n_features
        self.theta = np.zeros(n_features + 1) # weights (and bias)

    '''
    Loads a dataset. Specifically, returns a list of filenames, and dictionaries
    of classes and documents such that:
    classes[filename] = class of the document
    documents[filename] = feature vector for the document (use self.featurize)
    '''
    def load_data(self, data_set):
        filenames = []
        classes = dict()
        documents = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # your code here
                    # BEGIN STUDENT CODE
                    #using the three initialized lists/dicts I've appended the name of the file, 
                    # with the corresponding class/featrue vector
                    filenames.append(name)
                    classes.update({name : self.class_dict.get(os.path.basename(root))})
                    doc = []
                    for line in f:
                        for word in line.split():
                            doc.append(word)
                    documents.update({name : self.featurize(doc)})
    
                    # END STUDENT CODE
        return filenames, classes, documents

    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''
    def featurize(self, document):
        
        vector = np.zeros(len(self.feature_dict) + 1)
        vector[-1] = 1
        for word in document:
            if word in self.feature_dict:
                vector[self.feature_dict[word]] += 1
        # vector['doc_length'] = vector['doc_length'] / 10
        return vector

    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]                
                # create and fill in matrix x and vector y
                x = np.zeros((len(minibatch), self.n_features + 1))
                y = np.zeros((len(minibatch), ))

                for j in range(len(minibatch)):
                    x[j,:] = np.array(documents.get(minibatch[j]))
                    y[j] = classes.get(minibatch[j])
                y_hat = sigma(x.dot(self.theta))
                # update loss
                # print((1 - y).dot(np.log(1- y_hat)))
                loss += -((y.dot(np.log(y_hat))) + (1 - y).dot(np.log(1- y_hat)))
                # # compute gradient
                gradient = 1/len(minibatch) * ((x.T).dot(y_hat - y))
                #θt+1 = θt − η∇L.
                self.theta = self.theta - eta * gradient
                # END STUDENT CODE
            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = {}
        filenames, classes, documents = self.load_data(dev_set)
        for name in filenames:
            #results[filename][‘correct’] = correct class
            #results[filename][‘predicted’] = predicted class
            # BEGIN STUDENT CODE
            x = documents.get(name)
            y = sigma(x.dot(self.theta))
            
            if y > .5: 
                prediction = 1
            else: 
                prediction = 0
            
            results[name] = {'correct': classes.get(name), 'predicted':prediction}        

            # get most likely class (recall that P(y=1|x) = y_hat)
            # END STUDENT CODE
           
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
    
        precision = np.zeros(len(self.class_dict))
        recall = np.zeros(len(self.class_dict))
        F1 = np.zeros(len(self.class_dict)) 
        accuracy = np.zeros(len(self.class_dict))

        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        predicted = []
        correct = []
        
        for output in results:
            predicted.append(results[output]['predicted'])
            correct.append(results[output]['correct'])
        
        for actual, prediction in zip(correct, predicted):
            confusion_matrix[actual][prediction] += 1
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        
        for i in range(len(confusion_matrix)):
           
            row_sum = np.sum(confusion_matrix, axis=1)
            col_sum = np.sum(confusion_matrix, axis=0)
            
            if row_sum[i] == 0:
                precision[i] = 0
            else:
                precision[i] = confusion_matrix[i][i] / (row_sum[i])
            if col_sum[i] == 0:
                recall[i] = 0
            else:
                recall[i] = confusion_matrix[i][i] / (col_sum[i])
                
            if (precision[i] + recall[i]) != 0:
                F1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
                
        evaluation = {}
        class_i = 0
        for c in self.class_dict:
            evaluation[c] = {'Accuracy': accuracy, 'Precision': recall[class_i], 'Recall':precision[class_i], 'F1': F1[class_i]}
            class_i += 1
        final_eval = pd.DataFrame(evaluation)
        print(confusion_matrix)
        print(final_eval)


if __name__ == '__main__':
    lr = LogisticRegression(n_features=20)
    # make sure these point to the right directories
    lr.train('movie_reviews/train', batch_size= 4, n_epochs=100, eta=.001)
    #lr.train('movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
    results = lr.test('movie_reviews/dev')
    #results = lr.test('movie_reviews_small/test')
    lr.evaluate(results)

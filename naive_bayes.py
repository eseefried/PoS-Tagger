# CS114B Spring 2021 Programming Assignment 1
# Naive Bayes Classifier and Evaluation

from cmath import inf
import os
from select import select
import numpy as np
from collections import defaultdict
import pandas as pd

class NaiveBayes():

    def __init__(self):
        # be sure to use the right class_dict for each data set
        self.class_dict = {'neg': 0, 'pos': 1}
        # self.class_dict = {'action': 0, 'comedy': 1}
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
        self.prior = np.zeros(len(self.class_dict))
        self.likelihood = np.zeros((len(self.class_dict), len(self.feature_dict)))
        V = {}
        total_docs = 0
        class_docs = [None] * len(self.class_dict)
        big_doc = [None] * len(self.class_dict)
        class_path = ''
        class_i = -1
        
        # iterate over training documents
        for root, dirs, files in os.walk(train_set): 
            if self.class_dict.get(os.path.basename(root)) != None:
                doc_count = 0
                big_doc_temp = {}
                for name in files:
                    with open(os.path.join(root, name)) as f:
                        # collect class counts and feature counts
                        doc_count = len(files)
                        total_docs = len(name)
                        
                        if class_path != root: 
                            class_i += 1
                            if doc_count != 0: class_docs[class_i] = doc_count
                            self.prior[class_i] = np.log(class_docs[class_i] / total_docs)

                        class_path = root
                        
                        for line in f:   #iterate each line, creating a total V dict and seperate class dicts
                            for word in line.split():
                                if V.get(word) == None: 
                                    V.update({word: 1})
                                else: 
                                    V.update({word: V.get(word) + 1})
                                    
                                if big_doc_temp.get(word) == None: 
                                    big_doc_temp.update({word: 1})
                                else: 
                                    big_doc_temp.update({word: big_doc_temp.get(word) + 1})

                        if class_i >= 0: big_doc[class_i] = big_doc_temp           
            
                class_i = 0 
            
        #iterate over each class and fill out self likelihood
        if big_doc:
            for c in big_doc:
                for word in self.feature_dict:
                    count = 0 
                
                    if word in c: 
                        count += c[word] + 1
                    else: count = 1
                    bot = sum(c.values()) + len(V) 
                    self.likelihood[class_i][self.feature_dict.get(word)] =  np.log(count / bot)
                    
                    
                        
                class_i += 1
        print(self.prior)
        return self.likelihood, self.prior, V
             
        # normalize counts to probabilities, and take logs

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = {}
        # iterate over testing documents
                
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                if self.class_dict.get(os.path.basename(root)) != None:
                    with open(os.path.join(root, name)) as f:
                        # create feature vectors for each document
                        for line in f:
                            feat_vect = np.zeros(len(self.feature_dict))
                            for word in line.split():
                                if word in self.feature_dict:
                                    feat_vect[self.feature_dict.get(word)] += 1
                                                                        
                            log_likelihood = np.dot(self.likelihood, feat_vect)               
                            log_likelihood = log_likelihood + self.prior
                     
                            prediction = np.argmax(log_likelihood)
                            results[name] = {'correct': self.class_dict.get(os.path.basename(root)), 'predicted':prediction}

                # get most likely class

        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you may find this helpful

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
    
        # print(precision)
        # print(recall)
        # print(F1)
        evaluation = {}
        class_i = 0
        for c in self.class_dict:
            evaluation[c] = {'Accuracy': accuracy, 'Precision': precision[class_i], 'Recall':recall[class_i], 'F1': F1[class_i]}
            class_i += 1
        final_eval = pd.DataFrame(evaluation)
        print(final_eval)
            

    '''
    Performs feature selection.
    Returns a dictionary of features.
    '''
    def select_features(self, train_set):
        # almost any method of feature selection is fine here
        select_dict = {}
        select_iter = 0
        for root, dirs, files in os.walk(train_set):
            for name in files:
                if self.class_dict.get(os.path.basename(root)) != None:
                    with open(os.path.join(root, name)) as f:
                        # create feature vectors for each document
                        for line in f:
                            for word in line.split():
                                if select_dict.get(word) == None: 
                                    select_dict.update({word: 1})
                                    select_iter += 1
                                else: 
                                    select_dict.update({word: select_dict.get(word) + 1})
        
        num_features = int(np.ceil(len(select_dict) / 64))
               
        features = {}
        feature_number = 0
        # print(num_features)
        # while len(features) < num_features:
        # # for i in range(num_features):
        #     feature = max(select_dict, key=select_dict.get)
        #     if len(feature) > 1:
        #         features.update({feature: feature_number})  
        #         feature_number += 1
        #     select_dict.update({feature: 0})
        for i in range(num_features):
            feature = max(select_dict, key=select_dict.get) 
            select_dict[feature] = 0
            features.update({feature: i})  
     
        return features                      
        # return {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}

if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')
    # nb.train('movie_reviews_small/train')
    results = nb.test('movie_reviews/dev')
    # results = nb.test('movie_reviews_small/test')
    nb.evaluate(results)

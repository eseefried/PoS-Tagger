import os
import numpy as np
from collections import defaultdict
import random
from random import Random

class POSTagger():

    def __init__(self):
        # for testing with the toy corpus from worked example
        self.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
        self.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
                          'dwarf': 4, 'cheered': 5}
        # initial tag weights [shape = (len(tag_dict),)]
        self.initial = np.array([-0.3, -0.7, 0.3])
        # tag-to-tag transition weights [shape = (len(tag_dict),len(tag_dict))]
        self.transition = np.array([[-0.7, 0.3, -0.3],
                                    [-0.3, -0.7, 0.3],
                                    [0.3, -0.3, -0.7]])
        # tag emission weights [shape = (len(word_dict),len(tag_dict))]
        self.emission = np.array([[-0.3, -0.7, 0.3],
                                  [0.3, -0.3, -0.7],
                                  [-0.3, 0.3, -0.7],
                                  [-0.7, -0.3, 0.3],
                                  [0.3, -0.7, -0.3],
                                  [-0.7, 0.3, -0.3]])
        self.unk_index = -1

    '''
    Fills in self.tag_dict and self.word_dict, based on the training data.
    '''
    def make_dicts(self, train_set):
        tag_vocabulary = set()
        word_vocabulary = set()
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # BEGIN STUDENT CODE
                    # create vocabularies of every tag and word
                    #  that exists in the training data
                    for line in f:
                        for word in line.split():
                            i =  max(index for index, item in enumerate(word) if item == '/')
                            current_word = word[0:i]
                            current_tag = word[i+1:]
                            if current_tag not in tag_vocabulary:
                                tag_vocabulary.add(current_tag)
                            if current_word not in word_vocabulary:
                                word_vocabulary.add(current_word)     
                    # END STUDENT CODE
                    # remove pass keyword when finished
        # create tag_dict and word_dict
        # if you implemented the rest of this
        #  function correctly, these should be formatted
        #  as they are above in __init__
        self.tag_dict = {v: k for k, v in enumerate(tag_vocabulary)}
        self.word_dict = {v: k for k, v in enumerate(word_vocabulary)}

    '''
    Loads a dataset. Specifically, returns a list of sentence_ids, and
    dictionaries of tag_lists and word_lists such that:
    tag_lists[sentence_id] = list of part-of-speech tags in the sentence
    word_lists[sentence_id] = list of words in the sentence
    '''
    def load_data(self, data_set):
        sentence_ids = [] # doc name + ordinal number of sentence (e.g., ca010)
        sentences = dict()
        tag_lists = dict()
        word_lists = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    # BEGIN STUDENT CODE
                    # for each sentence in the document
                    #  1) create a list of tags and list of words that
                    #     appear in this sentence
                    #  2) create the sentence ID, add it to sentence_ids
                    #  3) add this sentence's tag list to tag_lists and word
                    #     list to word_lists
                    sentance_id = 0
                
                    for line in f:
                        sen = name + str(sentance_id)
                        sentence_ids.append(sen)
                        sentences[sen] = None
                        tags = []
                        words = []
                        for word in line.split():
                            # sentences[sen] += word
                            j =  max(index for index, item in enumerate(word) if item == '/')
                            current_word = word[0:j]
                            current_tag = word[j+1:]
                            
                            if current_word not in self.word_dict:
                                words.append(self.unk_index)
                            else:
                                words.append(self.word_dict[current_word])
                            
                            if current_tag not in self.tag_dict:
                                tags.append(self.unk_index)
                            else:
                                tags.append(self.tag_dict[current_tag])
                            
                            if sentences[sen] == None:
                                sentences.update({sen: word})
            
                            else:
                                sentences.update({sen: sentences[sen] + ' ' + word})
                            if word[-1] == '.':
                                sentance_id += 1
                        tag_lists.update({sen: tags})
                        word_lists.update({sen: words})
                    # END STUDENT CODE
                     # remove pass keyword when finished
        return sentence_ids, sentences, tag_lists, word_lists

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        best_path = []
        # BEGIN STUDENT CODE
        # initialization step
        #  fill out first column of viterbi trellis
        #  with initial + emission weights of the first observation
        v[:,0] = self.initial + self.emission[sentence[0]]
        # recursion step
        #  1) fill out the t-th column of viterbi trellis
        #  with the max of the t-1-th column of trellis
        #  + transition weights to each state
        #  + emission weights of t-th observateion
        #  2) fill out the t-th column of the backpointer trellis
        #  with the associated argmax values
        for t in range(1,T):
            v[:,t] = np.max(v[:,t-1] + self.transition[:,t-1] + self.emission[sentence[t]])
            backpointer[:,t] = np.argmax(v[:,t-1]  + self.transition[:,t-1] + self.emission[sentence[t]])
           
        print(backpointer) 
        # termination step
        #  1) get the most likely ending state, insert it into best_path
        best_path.append(np.argmax(v[:,-1]))
        for i in range(1,len(backpointer)+1):
            best_path.append(backpointer[best_path[i-1], i]) 
        best_path.reverse()
        #  2) fill out best_path from backpointer trellis
        # print(v[:,T])
        # best_path_prob = max(v[:,T])
        # END STUDENT CODE
        return best_path

    '''
    Trains a structured perceptron part-of-speech tagger on a training set.
    '''
    def train(self, train_set, dummy_data=None):
        self.make_dicts(train_set)
        sentence_ids, sentences, tag_lists, word_lists = self.load_data(train_set)
        if dummy_data is None: # for automated testing: DO NOT CHANGE!!
            Random(0).shuffle(sentence_ids)
            self.initial = np.zeros(len(self.tag_dict))
            self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
            self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        else:
            sentence_ids = dummy_data[0]
            sentences = dummy_data[1]
            tag_lists = dummy_data[2]
            word_lists = dummy_data[3]
        for i, sentence_id in enumerate(sentence_ids):
            # BEGIN STUDENT CODE
            # get the word sequence for this sentence and the correct tag sequence
            # use viterbi to predict
            # if mistake
            #  promote weights that appear in correct sequence
            #  demote weights that appear in (incorrect) predicted sequence
            # END STUDENT CODE
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of part-of-speech tags such that:
    results[sentence_id]['correct'] = correct sequence of tags
    results[sentence_id]['predicted'] = predicted sequence of tags
    '''
    def test(self, dev_set, dummy_data=None):
        results = defaultdict(dict)
        sentence_ids, sentences, tag_lists, word_lists = self.load_data(dev_set)
        if dummy_data is not None: # for automated testing: DO NOT CHANGE!!
            sentence_ids = dummy_data[0]
            sentences = dummy_data[1]
            tag_lists = dummy_data[2]
            word_lists = dummy_data[3]
        for i, sentence_id in enumerate(sentence_ids):
            # BEGIN STUDENT CODE
            # should be very similar to train function before mistake check
            # END STUDENT CODE
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return sentences, results

    '''
    Given results, calculates overall accuracy.
    This evaluate function calculates accuracy ONLY,
    no precision or recall calculations are required.
    '''
    def evaluate(self, sentences, results, dummy_data=False):
        if not dummy_data:
            self.sample_results(sentences, results)
        accuracy = 0.0
        # BEGIN STUDENT CODE
        # for each sentence, how many words were correctly tagged out of the total words in that sentence?
        # END STUDENT CODE
        return accuracy
        
    '''
    Prints out some sample results, with original sentence,
    correct tag sequence, and predicted tag sequence.
    This is just to view some results in an interpretable format.
    You do not need to do anything in this function.
    '''
    def sample_results(self, sentences, results, size=2):
        print('\nSample results')
        results_sample = [random.choice(list(results)) for i in range(size)]
        inv_tag_dict = {v: k for k, v in self.tag_dict.items()}
        for sentence_id in results_sample:
            length = len(results[sentence_id]['correct'])
            correct_tags = [inv_tag_dict[results[sentence_id]['correct'][i]] for i in range(length)]
            predicted_tags = [inv_tag_dict[results[sentence_id]['predicted'][i]] for i in range(length)]
            print(sentence_id,\
                sentences[sentence_id],\
                'Correct:\t',correct_tags,\
                '\n Predicted:\t',predicted_tags,'\n')

if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    pos.train('data_small/train') # train: toy data
    #pos.train('brown_news/train') # train: news data only
    #pos.train('brown/train') # train: full data
    # sentences, results = pos.test('data_small/test') # test: toy data
    #sentences, results = pos.test('brown_news/dev') # test: news data only
    #sentences, results = pos.test('brown/dev') # test: full data
    # print('\nAccuracy:', pos.evaluate(sentences, results))
    

from __future__ import print_function
import sys
import re
import pickle
import numpy as np

def load_file(filename):
    return open(filename, "r").read()

def max_num(text):
    n = 0
    for i in text.split("\n")[:-1]:
        n = max(n, int(i.split(" ")[0]))
    return n

def text2stories(text):
    n = max_num(text)
    text = re.sub('[^0-9a-z\s\n\t?]+', '', text.lower()).split("\n")
    stories = []
    story = []
    for i in text:
        if i == "":
            continue
        if i.split()[0] == "1":
            stories.append(story)
            story = []
        story.append(i)
#    stories = [text[i:i+n] for i in range(0, len(text), n)]
    sentences = []
    question = []
    answer = []
    for i in stories:
        story = []
        for j in i:
            if not "?" in j:
                story.append(j.split())
            else:
                sentences.append(list(story))
                question.append(j.split("\t")[0][:-1].split())
                answer.append(j.split("\t")[1])
    return sentences, question, answer

def build_dic(text):
    text = re.sub('[^0-9a-z\s\n\t]+', '', text.lower()).split()
    count = 1
    words = dict()
    words[""] = 0
    for i in text:
        if not i in words:
            words[i] = count
            count += 1
    return words

def text2array(text):
    sentences, question, answer = text2stories(text)
    max_num_facts = max_num(text)
    words = build_dic(text)
    words_dimension = len(words)
    max_len_sentences = max([len(j) for i in sentences for j in i]) + 1
    max_len_question = max([len(i) for i in question])
    array_sentences = np.zeros([len(sentences), max_num_facts, max_len_sentences], dtype = np.int32)
    array_question = np.zeros([len(question), max_len_question], dtype = np.int32)
    array_answer = np.zeros([len(answer), words_dimension], dtype = np.int32)
    
    for i in range(len(sentences)):
        for j in range(len(question[i])):
            array_question[i, j] = words[question[i][j]]
        for j in range(len(sentences[i])):
            for k in range(len(sentences[i][j])):
                array_sentences[i, j, k] = words[sentences[i][j][k]]
        array_answer[i, words[answer[i]]] = 1
        
    return array_sentences, array_question, array_answer, words

def print_task(array, question, answer, words):
    reversed_words = {v: k for k, v in words.items()}
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if reversed_words[array[i][j][0]] == "":
                continue
            for k in range(array.shape[2]):
                print(reversed_words[array[i][j][k]], end = " ")
            print("\n")
        for j in range(question.shape[1]):
            print(reversed_words[question[i][j]], end = " ")
        print("?", end = '      ')
        for j in range(answer.shape[1]):
            if answer[i][j] == 1:
                print(reversed_words[j])

def print_sentences(array, words):
    reversed_words = {v: k for k, v in words.items()}
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                print(reversed_words[array[i][j][k]], end = " ")
            print("\n")  
            
            
def word2num(words, i):
    return words[i]

def num2word(words, i):
    reversed_words = {v: k for k, v in words.items()}
    return reversed_words[i]

class BatchGenerator:
    def __init__(self, filename, batch_size, validation_size):
        text = load_file(filename)
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.pos = 0
        sentences, question, answer, self.words = text2array(text)
        self.sentences, self.question, self.answer = sentences[validation_size:, :, :], question[validation_size:, :], answer[validation_size:, :]
        self.validation = (sentences[:validation_size, :, :], question[:validation_size,:], answer[:validation_size, :])
                           
    def get_validation(self):
        return list(self.validation)
        
    def get_next_batch(self, batch_size = None):
        if batch_size == None:
            batch_size = self.batch_size
        if (self.pos + batch_size > self.sentences.shape[0]):
            self.pos = 0
        batch = (self.sentences[self.pos : self.pos + batch_size, :, :], self.question[self.pos : self.pos + batch_size, :], self.answer[self.pos : self.pos + batch_size, :])
        self.pos += batch_size
        return batch
            
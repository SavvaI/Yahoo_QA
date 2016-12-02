import pickle
import numpy as np
import random

class Batch_Generator:
    def __init__(self, batch_size, mode, num_answers):
        self.num_answers = num_answers
        self.batch_size = batch_size
        self.input_file = open("data/yahoo_qa.txt", 'r')
        self.embedding_file = open("data/embeddings.txt", "r")
        self.num = iter(map(int, open("data/embeddings.txt", "r").read().split()))
        self.mode = mode

    def read_embedding(self):
        return pickle.load(self.embedding_file)

    def sum_answer(self):
        embed_sum = np.zeros((1, 1024))
        nxt = self.num.next()
        print(nxt)
        for i in range(self.num.next()):
            embed_sum += self.read_embedding()
        return embed_sum

    def gen_sum_batch(self):
        answer_batch = np.zeros((self.batch_size, self.num_answers, 1024))
        question_batch = np.zeros((self.batch_size, 1024))
        _y_batch = np.array([random.randint(0, self.num_answers - 1) for i in range(self.batch_size)])

        for i in range(self.batch_size):
            for j in range(self.num_answers):
                question = self.read_embedding()
                answer = self.sum_answer()
                answer_batch[i, j] = answer[0]
                if j == _y_batch[i]:
                    question_batch[i] = question[0]

        return (answer_batch, question_batch, _y_batch)

    def gen_next_batch(self):
        batch = []
        for i in range(self.batch_size * 2):
            question_sentence = ''
            c = self.input_file.read(1)
            if c == '':
                self.input_file = open("data/yahoo_qa.txt", 'r')
            while c != '>' and c != '':
                question_sentence += c
                c = self.input_file.read(1)
            batch.append(question_sentence)
        return batch

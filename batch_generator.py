import pickle
import numpy as np
import random
import re

class BatchGenerator:
    def __init__(self, batch_size, mode = 'sum', num_variants = 2, encoding_length = 20):
        self.num_variants = num_variants
        self.batch_size = batch_size
        self.input_file = open("data/yahoo_qa.txt", 'r')
        self.embedding_file = open("data/embeddings.txt", "rb")
        self.num = iter(map(int, open("data/num.txt", "r").read().split()))
        self.mode = mode
        self.encode_dict = pickle.load(open("data/encode_dict.txt", 'rb'))
        self.encoding_length = encoding_length
        self.num_questions = 285254 / 2

    def encode_word(self, word):
        if word.lower() in self.encode_dict:
            return self.encode_dict[word.lower()]
        else:
            return self.encode_dict['']

    def count_questions(self):
        cnt = 0
        while True:
            c = self.input_file.read(1)
            if c == '':
                break
            while c != '>':
                c = self.input_file.read(1)
            cnt += 1
        return cnt


    def read_qa(self):
        #Reads sentences from text file, until meets > symbol
        c = self.input_file.read(1)
        if c == '':
            self.input_file.seek(0)
            self.input_file.read(1)
        sentences = ''
        while c != '>':
            sentences += c
            c = self.input_file.read(1)
        sentences = sentences.replace('\n', '')
        return sentences

    def get_text_batch(self, encode = False, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        #Gets batch from raw txt file, encodes words if 'encode' flag is on
        questions = [[] for i in range(batch_size)]
        answers = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(self.num_variants * 2):
                if j % 2 == 0:
                    questions[i].append(self.read_qa())
                else:
                    answers[i].append(self.read_qa())

        if encode:
            _y = np.array([random.randint(0, self.num_variants - 1) for i in range(batch_size)])
            encoded_questions = np.zeros((batch_size, self.encoding_length))
            encoded_answers = np.zeros((batch_size, self.num_variants, self.encoding_length))
            for i in range(batch_size):
                for j in range(self.num_variants):
                    a = np.array(list(map(self.encode_word, answers[i][j].split())))[:self.encoding_length]
                    encoded_answers[i, j, :len(a)] = a[:]
#                    encoded_questions[i, j] = np.array(map(self.encode_word, questions[i][j].split()))
                q = np.array(list(map(self.encode_word, questions[i][_y[i]].split())))[:self.encoding_length]
                encoded_questions[i, :len(q)] = q[:]
            return answers, questions, encoded_answers, encoded_questions, _y

        else:
            return answers, questions

    def read_embedding(self):
        #Reads one embedding from embedding file
        return pickle.load(self.embedding_file)

    def sum_answer(self):
        #Sums answer sentences embeddings with respect to number given in num_file
        embed_sum = np.zeros((1, 1024))
        nxt = self.num.next()
        #print(nxt)
        for i in range(nxt):
            embed_sum += self.read_embedding()
        return embed_sum


    def get_sum_batch(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        answer_batch = np.zeros((batch_size, self.num_variants, 1024))
        question_batch = np.zeros((batch_size, 1024))
        _y_batch = np.array([random.randint(0, self.num_variants - 1) for i in range(batch_size)])

        for i in range(batch_size):
            for j in range(self.num_variants):
                question = self.read_embedding()
                answer = self.sum_answer()
                answer_batch[i, j] = answer[0]
                if j == _y_batch[i]:
                    question_batch[i] = question[0]

        return (answer_batch, question_batch, _y_batch)



    def next_batch(self):
        if self.mode == "sum":
            return self.gen_sum_batch()


# g = BatchGenerator(100, 'sum', 2)
# print(g.num_questions)
# cnt = 0
# while True:
#     batch = g.get_text_batch()
#     if cnt % 100 == 0:
#         print(str(cnt * g.num_variants * g.batch_size) + '/' + '142627')
#     cnt += 1
#     if (cnt) * g.num_variants * g.batch_size > 1:
#         print(batch[1][0][0], batch[0][0][0])

# count = 0
# while True:
#     count += 1
#
# print(count)


# for i in batch[0]:
#     for j in i:
#         print(j)
#         print(np.array(list(map(g.encode_word, j.split()))))
#     print()

#print(g.encode_dict.items())
# for i in range(1):
#     print(g.get_text_batch())

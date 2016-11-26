class Batch_Generator:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.input_file = open("data/yahoo_qa.txt", 'r')
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

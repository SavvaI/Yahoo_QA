import numpy as np
import random
import pickle

size = 285254 / 2
train_size = int(size * 0.95)
validation_size = size - train_size

#Splitting dataset
embeddings = open("data/embeddings_shuffled.txt", "rb")
num = open("data/num_shuffled.txt", "rb")
text = open("data/yahoo_qa_shuffled.txt", "rb")
numbers = pickle.load(open("data/num_shuffled.txt", "rb"))
# numbers = map(int, open("data/num_shuffled.txt", "r").read().split())
validation_emb = open("data/validation_embeddings_shuffled.txt", "wb")
validation_num = open("data/validation_num_shuffled.txt", "wb")
validation_text = open("data/validation_text_shuffled.txt", "wb")
train_emb = open("data/train_embeddings_shuffled.txt", "wb")
train_num = open("data/train_num_shuffled.txt", "wb")
train_text = open("data/train_text_shuffled.txt", 'wb')

# print(size == len(numbers), validation_size + train_size == size)

for i in range(train_size):
    pickle.dump(pickle.load(embeddings), train_emb)
    pickle.dump(pickle.load(text), train_text)
pickle.dump(numbers[:train_size], train_num)


for i in range(validation_size ):
    pickle.dump(pickle.load(embeddings), validation_emb)
    pickle.dump(pickle.load(text), validation_text)
pickle.dump(numbers[train_size:], validation_num)



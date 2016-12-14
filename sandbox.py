import numpy as np
import random
import pickle

size = 285254 / 2
train_size = int(size * 0.95)
validation_size = size - train_size

#Splitting dataset
embeddings = open("data/embeddings.txt", "rb")
num = open("data/num.txt", "r")
text = open("data/yahoo_qa.txt").read()
numbers = map(int, open("data/num.txt", "r").read().split())
validation_emb = open("data/validation_embeddings.txt", "wb")
validation_num = open("data/validation_num.txt", "w")
validation_text = open("data/validation_text.txt", "w")
train_emb = open("data/train_embeddings.txt", "wb")
train_num = open("data/train_num.txt", "w")
train_text = open("data/train_text.txt", 'w')
text = text.split(">")

print(size == len(numbers), validation_size + train_size == size)

train_text.write('>'.join(text[:train_size*2]))
for i in range(train_size):
    train_num.write(str(numbers[i]) + " ")
    emb = pickle.load(embeddings)
    pickle.dump(emb, train_emb)
    for j in range(numbers[i]):
        emb = pickle.load(embeddings)
        pickle.dump(emb, train_emb)


validation_text.write('>'.join(text[train_size*2:train_size*2+validation_size*2]))
for i in range(train_size, train_size + validation_size):
    validation_num.write(str(numbers[i]) + " ")
    emb = pickle.load(embeddings)
    pickle.dump(emb, validation_emb)
    for j in range(numbers[i]):
        emb = pickle.load(embeddings)
        pickle.dump(emb, validation_emb)



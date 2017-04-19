import numpy as np
import numpy.random as random
import os
import pickle
path = os.path.dirname(os.path.abspath(__file__))


def shuffle():
    embeddings_file = open(path + "/data/embeddings.txt", "rb")
    embeddings = []
    # while 1:
    #     try:
    #         embeddings.append(pickle.load(embeddings_file))
    #     except EOFError:
    #         break

    num = map(int, open(path + "/data/num.txt", "r").read().split(' ')[0:-1])
    num = np.array(num)
    text = open(path + "/data/yahoo_qa.txt").read().split(">")
    perm = random.permutation(range(len(text) / 2))
    shuffled_text_file = open(path + "/data/yahoo_qa_shuffled.txt", 'wb')
    shuffled_embeddings_file = open(path + "/data/embeddings_shuffled.txt", "wb")
    shuffled_num_file = open(path + "/data/num_shuffled.txt", "wb")

    print(num)
    print(len(embeddings), len(text) / 2, len(num), len(perm))
    for i in range(len(perm)):
        pickle.dump((text[perm[i] * 2], text[perm[i] * 2 + 1]), shuffled_text_file)
        # pickle.dump(text[perm[i] * 2 + 1], shuffled_text_file)

    text = None

    for i in range(len(num)):
        embeddings.append([])
        embeddings[-1].append(pickle.load(embeddings_file))
        embeddings[-1].append([])
        for j in range(num[i]):
            embeddings[-1][1].append(pickle.load(embeddings_file))

    for i in range(len(perm)):
        pickle.dump(embeddings[perm[i]], shuffled_embeddings_file)
        # pickle.dump(embeddings[perm[i]][0], shuffled_embeddings_file)
        # for j in embeddings[perm[i]][1]:
        #     pickle.dump(j, shuffled_embeddings_file)

    embeddings = None
    num_shuffled = np.ndarray(dtype=np.int32, shape=[len(num)])
    for i in range(len(perm)):
        num_shuffled[i] = num[perm[i]]

    pickle.dump(num_shuffled, shuffled_num_file)

def check():
    shuffled_text_file = open(path + "/data/yahoo_qa_shuffled.txt", 'rb')
    shuffled_embeddings_file = open(path + "/data/embeddings_shuffled.txt", "rb")
    shuffled_num_file = open(path + "/data/num_shuffled.txt", "rb")
    num = pickle.load(shuffled_num_file)
    for i in range(len(num)):
        emb = pickle.load(shuffled_embeddings_file)
        if num[i] != len(emb[1]):
            print "Error"

shuffle()
check()


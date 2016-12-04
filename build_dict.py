import pickle
DICT_SIZE = 500
text = open('data/yahoo_qa.txt', 'r').read()
words = text.lower().replace('>', '').split()
count_dict = {}

for i in words:
    if not i in count_dict:
        count_dict[i] = 1
    else:
        count_dict[i] += 1

items = list(count_dict.items())
items.sort(key=lambda x: x[1])
items = items[::-1]
# print(items[:200])
# for i in range(len(items)):
#      if items[i][1] < 500:
#          break
items = items[:DICT_SIZE]
print(items[0][1], "-", items[-1][1])

encode_dict = {items[i][0]: i + 1 for i in range(len(items))}
encode_dict[''] = 0
print(len(encode_dict))

pickle.dump(encode_dict, open('data/encode_dict.txt', 'wb'))



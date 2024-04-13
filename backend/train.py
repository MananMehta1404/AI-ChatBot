import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load the json file
with open('knowledge_base.json', 'r') as f:
    intents = json.load(f)

all_words = []   # List of all words from our intents patterns
tags = []        # List of all tags
xy = []          # List of patterns and tags

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']

    # Add to tag list
    tags.append(tag)

    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = tokenize(pattern)

        # Add to our words list
        all_words.extend(w)

        # Add to xy pair
        xy.append((w, tag))

# Stem and lower each word and ignore the punctuation charcaters if any
ignore_words = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Remove duplicates and sort the words
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# print(X_train)
# print(y_train)

# Create PyTorch dataset
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # We can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)
import json

from nltk_utils import tokenize, stem

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

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)
import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# Tokenization
def tokenize(sentence):
    """
    Tokenize: Split sentence into array of words/tokens
    Token: A token can be a word, a punctuation character, or a number
    """

    return nltk.word_tokenize(sentence)

# sentence = "Hey, how are you doing today?"
# tokenized_sentence = tokenize(sentence)
# print(tokenized_sentence)

# Stemming
def stem(word):
    """
    Stemming: Find the root form of the word
    Examples:
        words = ["run", "running", "runs"]
        words = [stem(w) for w in words]
        -> ["run", "run", "run"]
    """

    return stemmer.stem(word.lower())

# words = ["run", "running", "runs"]
# words = [stem(w) for w in words]
# print(words)

# Bag of words
def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array: 1 for each known word that exists in the sentence, otherwise 0
    example:
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """

    # Stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]

    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

# all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool", "how"]
# bag = bag_of_words(tokenized_sentence, all_words)
# print(bag)
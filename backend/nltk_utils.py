import nltk
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

sentence = "Hey, how are you doing today?"
tokenized_sentence = tokenize(sentence)
print(tokenized_sentence)

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

words = ["run", "running", "runs"]
words = [stem(w) for w in words]
print(words)
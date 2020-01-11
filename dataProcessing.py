
import pandas as pd

import numpy as np
import re
import nltk
from nltk.corpus import wordnet
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


def preProcessing(pathToData):
    data = pd.read_csv(pathToData)
    data.drop('id', axis=1, inplace=True)

    # print(data.head())


    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, nltk.wordnet.NOUN)


    stemmer = WordNetLemmatizer()
    print(len(data.comments))

    def postCleaner(post):
        # Remove hyperlinks
        processedPost = re.sub(r'https?:\/\/.*\/\w*', '', str(post))

        # remove all special characters
        processedPost = re.sub(r'\W+', ' ', processedPost)

        # remove all single characters from the start
        # processedPost = re.sub(r'\^[a-zA-Z]\s+', ' ', processedPost)

        #remove numbers
        processedPost = re.sub('\d+', ' ', processedPost)

        # Converting to lowercase
        processedPost = processedPost.lower()

        # Substituting multiple spaces with single space
        processedPost = re.sub(r'\s+', ' ', processedPost, flags=re.I)

        # lemmatization and delete word with length less than 4
        processedPost = [stemmer.lemmatize(word, get_wordnet_pos(word)) for word in nltk.word_tokenize(processedPost) if len(word)>=3]
        processedPost = ' '.join(processedPost)

        # remove all single characters
        processedPost = re.sub(r'\s+[a-zA-Z]\s+', '', processedPost)

        return processedPost

    data.comments = data.comments.apply(postCleaner)

    return data

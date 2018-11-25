"""
This function returns features of a question.
The maximum length of question is 54
# 54 for trianing set
# 41 for test set
Zero Padding is done for shorter questions.
The size of each word embedding is 300
A tensor (1,54,300) is returned
"""

import spacy
import numpy as np
# import en_core_web_lg # Use this one
import en_core_web_md

max_length_questions = 54
word_feature_size = 300

def get_question_features(question):
    word_embeddings = spacy.load('en_core_web_md')
    tokens = word_embeddings(question)
    question_feature = np.zeros((1,54,300)) # Use max_length_questions here instead of 54 and word_feature_size instead of 300
    for i in range(len(tokens)):
        question_feature[0,i,:] = tokens[i].vector
    return question_feature

import numpy as np
import re

from statistics import mean

import spacy

nlp = spacy.load('en_core_web_lg')

def get_index(string, substring):
    try:
        if re.search(r"\b{}\b".format(substring), string):
            i = string.index(substring)
        else:
            i = -1
    except:
        i = -1

    return i


def index_ent_in_str(ent, doc_text):
    ent_first_index = []
    ent_last_index = []
    ent_i_first_word = []
    ent_i_last_word = []
    doc_text_substr = doc_text
    ent_in_substring = get_index(doc_text_substr, ent) != -1
    i = 0
    while ent_in_substring:
        i_first = get_index(doc_text_substr, ent)
        i_last = i_first + len(ent) - 1

        if i_first != 0:
            number_of_word_before = doc_text[:i_first + i - 1].count(" ") + 1
        else:
            number_of_word_before = 0

        number_of_word_in = doc_text_substr[i_first:i_last + 1].count(" ")

        doc_text_substr = doc_text_substr[i_last + 1:]
        ent_first_index.append(i_first + i)
        ent_last_index.append(i_last + i)
        ent_i_first_word.append(number_of_word_before + 1)
        ent_i_last_word.append(number_of_word_before + number_of_word_in + 1)

        i = i_last + i + 1
        ent_in_substring = get_index(doc_text_substr, ent) != -1

    return ent_i_first_word, ent_i_last_word


def custom_distance(query, ent, doc_text):
    doc_text = doc_text.lower()
    ent = ent.lower()
    query = query.replace(" AND ", " ").replace(" OR ", " ").replace("(", "").replace(")", "").lower()
    ent_i_first_word, ent_i_last_word = index_ent_in_str(ent, doc_text)

    query_i_words = []
    for q in query.split(" "):
        q_modif = q.replace("*", "")
        query_i_words.append(index_ent_in_str(q_modif, doc_text)[0])

    dist_all_ent = []
    for i in range(len(ent_i_first_word)):
        dist_to_ent_i = []
        for word in query_i_words:
            if len(word) != 0:
                dist_to_word = []
                for index in word:
                    first_word_dist = abs(ent_i_first_word[i] - index)
                    last_word_dist = abs(ent_i_last_word[i] - index)
                    dist_to_word.append(min(first_word_dist, last_word_dist))
                dist_to_ent_i.append(min(dist_to_word))
        if len(dist_to_ent_i) != 0:
            dist_all_ent.append(mean(dist_to_ent_i))

    if len(dist_all_ent) == 0:
        return float('inf')
    else:
        return min(dist_all_ent)


def final_answer(query, answers, distances, n=2):
    score_answer = {}
    freq_answer = {}
    for i, answer in enumerate(answers):
        if answer in score_answer:
            score_answer[answer] += distances[i]
            freq_answer[answer] += 1
        else:
            score_answer[answer] = distances[i]
            freq_answer[answer] = 1

    query = query.replace(" AND ", " ").replace(" OR ", " ").replace("(", "").replace(")", "").lower()

    for key in score_answer.keys():
        score_answer[key] = (score_answer[key] / answers.count(key) ** (7/2))
        add_penalty = True
        doc = nlp(key)
        for token in doc:
            if token.pos_ not in ["PRON", "ADP", "AUX", "CCONJ", "CONJ", "DET", "PUNCT", "SCONJ", "PART", "ADV"]\
                    and token.text not in query:
                add_penalty = False
        if add_penalty:
            score_answer[key] += 10

    answers = []
    n = min(n, len(score_answer))
    for i in range(n):
        answer = min(score_answer, key=score_answer.get)
        del score_answer[answer]
        answers.append(answer)

    return answers

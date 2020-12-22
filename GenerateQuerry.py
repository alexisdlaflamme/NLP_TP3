from nltk.stem.porter import *
import spacy

nlp = spacy.load('en_core_web_lg')
stemmer = PorterStemmer()


# Conserver seulement les mots qui ne sont pas des mots outils, de verbe, mots de questions, poctuation. POS tagging
# Si aucune retirer les mots les moins utiles pour classer la question.

def generate_query(sentence):
    doc = nlp(sentence)
    query = []
    for token in doc:
        if token.pos_ not in ["PRON", "ADP", "AUX", "CCONJ", "SCONJ", "CONJ", "DET", "PUNCT", "PART", "ADV"] and \
                token.text.lower() not in ["how", "why", "whose", "whom", "who", "which", "where", "when", "what", "much", "many"]:
            if token.pos_ not in ["PROPN", "NOUN", "NUM"]:
                query.append("(" + stemmer.stem(token.text) + "*" + " OR ")
                query.append(token.lemma_ + " OR ")
                query.append(token.text + ") AND ")
            else:
                query.append(token.text + " AND ")
    query_string = "".join(query)
    return query_string[:-5]


if __name__ == "__main__":
    print(generate_query("What did Carl Yung studied?"))

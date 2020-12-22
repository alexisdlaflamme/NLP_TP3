import os

import spacy
from whoosh import scoring, qparser
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from QuestionClassifier import QuestionClassifier, load_question_classifier, obtain_prediction
from GenerateQuerry import generate_query
from createSearchableData import createSearchableData
from generateRespond import final_answer, custom_distance

if __name__ == "__main__":

    print("###########################")
    print("#  Bienvenue dans QA 1.0  #")
    print("###########################")

    if not os.path.exists('indexdir'):
        print("")
        print("Veuillez patienter pendant la l'indexation du corpus")
        createSearchableData("./corpus")
        print("Terminé! Vous pourrez utiliser notre système Q&A dans quelques secondes")

    print("")
    print("Veuillez faire attention à votre ortographe lorsque vous posez une question!")
    print("")

    model = load_question_classifier()
    ix = open_dir("indexdir")
    nlp = spacy.load('en_core_web_lg')

    done = False
    while not done:
        # Get the question
        print("Poser une question en anglais (Entrez 0 pour terminer) : ")

        question = ""
        while not question:
            question = str(input())
            if not question:
                print("*** La question ne peut pas être une string vide ***")
                print("Poser une question en anglais (Entrez 0 pour terminer) : ")

        if question == "0":
            done = True
            break

        print("Question : {}".format(question))

        # Create query for the question
        question_query = generate_query(question)
        print("Query : {}".format(question_query))

        # Find type of question
        question_type = obtain_prediction(model, question, spacy_ent=True)
        qt_not_spacy = obtain_prediction(model, question)
        print("Question type : {} ({})".format(qt_not_spacy, question_type))

        list_answer = []
        liste_dist = []
        with ix.searcher(weighting=scoring.Frequency) as searcher:
            parser = QueryParser("content", ix.schema)
            parser.add_plugin(qparser.WildcardPlugin())
            query = parser.parse(question_query)
            results = searcher.search(query, limit=5)
            if len(results.top_n) > 0:
                for i in range(len(results.top_n)):
                    for ent in nlp(results[i]['textdata']).ents:
                        if ent.label_ in question_type or len(question_type) == 0:
                            list_answer.append(ent.text.lower())
                            liste_dist.append(custom_distance(question_query, ent.text, results[i]['textdata']))
                answers = final_answer(question_query, list_answer, liste_dist)
                for i_ans in range(len(answers)):
                    print('Réponse {} : {}'.format(i_ans, answers[i_ans]))
            else:
                print("La réponse est introuvable!")

            print("")

    print("################")
    print("#  Au revoir!  #")
    print("################")

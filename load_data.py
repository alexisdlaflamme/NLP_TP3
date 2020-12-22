from nltk import word_tokenize


def load_question_type():
    questions_train = open("data/train_TREC.txt", 'r')
    questions_test = open("data/test_TREC.txt", 'r')

    questions_train = [line for line in questions_train.readlines()]
    questions_test = [line for line in questions_test.readlines()]

    x_train, y_train = split_xy(questions_train)
    x_test, y_test = split_xy(questions_test)

    return x_train, y_train, x_test, y_test


def split_xy(questions):
    x = []
    y = []
    for question in questions:
        i = question.find(" ") + 1
        j = question.find(":")
        x.append(question[i:])
        y.append(question[:j])

    return x, y

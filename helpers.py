import pickle


def save_item_to_pkl(path, item):
    with open(path, 'wb') as output:
        pickle.dump(item, output, pickle.HIGHEST_PROTOCOL)


def load_item_from_pkl(path):
    with open(path, 'rb') as input:
        item = pickle.load(input)
    return item


class model_param:
    def __init__(self, embedding_layer, hidden_size, nb_class):
        self.embedding_layer = embedding_layer
        self.hidden_size = hidden_size
        self. nb_class = nb_class
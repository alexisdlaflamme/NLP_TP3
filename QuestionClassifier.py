import torch
import spacy
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from poutyne.framework import Experiment
from helpers import save_item_to_pkl, model_param, load_item_from_pkl
from load_data import load_question_type
from QuestionDataset import QuestionDataSet, pad_batch
from pandas_ml import ConfusionMatrix

NEG_INF = -1e6
nlp = spacy.load('en_core_web_lg')


class QuestionClassifier(nn.Module):

    def __init__(self, embedding, hidden_state_size, nb_class):
        super().__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(embedding)
        embedding_size = embedding.size()[1]
        self.rnn = nn.LSTM(embedding_size, hidden_state_size, 1, bidirectional=True)
        self.attention_layer = nn.Linear(2 * hidden_state_size, 1)  # On calcule un facteur (scalaire) par input
        self.classification_layer = nn.Linear(2 * hidden_state_size, nb_class)  # 2 * -> Une pour chaque direction

    def forward(self, x, x_lenghts):
        x = self.embedding_layer(x)
        x = self._handle_rnn_output(x, x_lenghts)
        x = self.classification_layer(x)

        return x

    def _handle_rnn_output(self, x, x_lenghts):
        packed_batch = pack_padded_sequence(x, x_lenghts, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(packed_batch)
        unpacked_rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
        sequence_mask = self.make_sequence_mask(x_lenghts)
        attention = self.attention_layer(unpacked_rnn_output)
        soft_maxed_attention = self.mask_softmax(attention.squeeze(-1), sequence_mask)
        attention_weighted_rnn_output = torch.sum(soft_maxed_attention.unsqueeze(-1) * unpacked_rnn_output, dim=1)

        return attention_weighted_rnn_output

    def calculate_attention_for_input(self, x, x_lenghts):
        x = self.embedding_layer(x)
        packed_batch = pack_padded_sequence(x, x_lenghts, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(packed_batch)
        unpacked_rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
        sequence_mask = self.make_sequence_mask(x_lenghts)
        attention = self.attention_layer(unpacked_rnn_output)
        soft_maxed_attention = self.mask_softmax(attention.squeeze(-1), sequence_mask)
        return soft_maxed_attention

    @staticmethod
    def make_sequence_mask(sequence_lengths):
        maximum_length = torch.max(sequence_lengths)

        idx = torch.arange(maximum_length).to(sequence_lengths).repeat(sequence_lengths.size(0), 1)
        mask = torch.gt(sequence_lengths.unsqueeze(-1), idx).to(sequence_lengths)

        return mask

    @staticmethod
    def mask_softmax(matrix, mask=None):
        if mask is None:
            result = nn.functional.softmax(matrix, dim=-1)
        else:
            mask_norm = ((1 - mask) * NEG_INF).to(matrix)
            for i in range(matrix.dim() - mask_norm.dim()):
                mask_norm = mask_norm.unsqueeze(1)
            result = nn.functional.softmax(matrix + mask_norm, dim=-1)

        return result


def obtain_prediction(model, sentence, spacy_ent=False):
    idData2idSpacy = {"ABBR": ["ORG", "EVENT", "WORK_OF_ART", "LANGUAGE", "LAW"],
                      "DESC": [],
                      "ENTY": ["ORG", "EVENT", "LANGUAGE", "PRODUCT", "MONEY", "NORP"],
                      "HUM": ["PERSON", "ORG"],
                      "LOC": ["FAC", "GPE", "LOC"],
                      "NUM": ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]}

    word2id = load_item_from_pkl("model/word2id.pkl")
    id2lable = load_item_from_pkl("model/id2lable.pkl")

    tokenized_sentence = [word2id.get(word.text, 1) for word in nlp(sentence)]
    sentence_length = len(tokenized_sentence)
    class_score = model(torch.LongTensor(tokenized_sentence).unsqueeze(0),
                        torch.LongTensor([sentence_length])).detach().numpy()

    label = id2lable[np.argmax(class_score)]
    if spacy_ent:
        return idData2idSpacy[label]
    else:
        return label


def obtain_attention(model, sentence):
    word2id = load_item_from_pkl("model/word2id.pkl")

    tokenized_sentence = [word2id.get(word.text, 1) for word in nlp(sentence)]
    sentence_length = len(tokenized_sentence)
    attention = model.calculate_attention_for_input(torch.LongTensor(tokenized_sentence).unsqueeze(0),
                                                    torch.LongTensor([sentence_length])).squeeze(0).detach().numpy()
    return list(zip(nlp(sentence), attention))


def evaluate(model, x, y):
    prediction = obtain_prediction(model, x)

    print("\nQ: {}. \nPred: {}, Truth: {}".format(x, prediction, y))


def load_question_classifier():
    info_model = load_item_from_pkl("model/info_model.pkl")
    model = QuestionClassifier(info_model.embedding_layer, info_model.hidden_size, info_model.nb_class)
    experiment = Experiment('model/QuestionClassifier',
                            model,
                            optimizer="Adam",
                            task="classification")
    experiment.load_checkpoint(checkpoint="best")

    return model


if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid = load_question_type()

    # On converti les label textuels en index numérique
    label = list(set(y_train))
    label.sort()
    id2lable = {label_id: value for label_id, value in enumerate(label)}
    label2id = {value: label_id for label_id, value in id2lable.items()}

    y_train = [label2id[label] for label in y_train]
    y_valid = [label2id[label] for label in y_valid]

    nb_class = len(id2lable)

    nlp = spacy.load('en_core_web_lg')
    embedding_size = nlp.meta['vectors']['width']

    word2id = {}
    id2embedding = {}

    word2id[1] = "<unk>"
    id2embedding[1] = np.zeros(embedding_size, dtype=np.float64)

    word_index = 2

    for question in X_train:
        for word in nlp(question):
            if word.text not in word2id.keys():
                word2id[word.text] = word_index
                id2embedding[word_index] = word.vector
                word_index += 1

    save_item_to_pkl("model/word2id.pkl", word2id)
    save_item_to_pkl("model/id2embedding.pkl", id2embedding)
    save_item_to_pkl("model/id2lable.pkl", id2lable)
    save_item_to_pkl("model/label2id.pkl", label2id)

    train_dataset = QuestionDataSet(X_train, y_train, word2id, nlp)
    valid_dataset = QuestionDataSet(X_valid, y_valid, word2id, nlp)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_batch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True, collate_fn=pad_batch)

    id2embedding[0] = np.zeros(embedding_size, dtype=np.float32)
    embedding_layer = np.zeros((len(id2embedding), embedding_size), dtype=np.float32)
    for token_index, embedding in id2embedding.items():
        embedding_layer[token_index, :] = embedding
    embedding_layer = torch.from_numpy(embedding_layer)

    hidden_size = 250
    model_info = model_param(embedding_layer, hidden_size, nb_class)
    save_item_to_pkl("model/info_model.pkl", model_info)

    model = QuestionClassifier(embedding_layer, hidden_size, nb_class)
    experiment = Experiment('model/QuestionClassifier',
                            model,
                            optimizer="Adam",
                            task="classification")
    # logging = experiment.train(train_dataloader, valid_dataloader, epochs=15, disable_tensorboard=True)
    experiment.load_checkpoint(checkpoint="best")

    y_pred = []
    y_true = []
    for i in range(len(y_valid)):
        y_pred.append(obtain_prediction(model, X_valid[i]))
        y_true.append(id2lable[y_valid[i]])

    cm = ConfusionMatrix(y_true, y_pred)
    print(cm)

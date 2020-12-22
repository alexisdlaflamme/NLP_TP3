import torch
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple
import numpy as np


class QuestionDataSet(Dataset):
    def __init__(self, dataset: List[str], target: np.array, word2id: Dict[str, int], nlp_model):
        self.tokenized_dataset = [None for _ in range(len(dataset))]
        self.dataset = dataset
        self.target = target
        self.word2id = word2id
        self.nlp_model = nlp_model

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.tokenized_dataset[index] is None:
            self.tokenized_dataset[index] = self.tokenize(self.dataset[index])

        return LongTensor(self.tokenized_dataset[index]), LongTensor([self.target[index]]).squeeze(0)

    def tokenize(self, sentence):
        return [self.word2id.get(word.text, 1) for word in self.nlp_model(sentence)]


def pad_batch(batch: List[Tuple[LongTensor, LongTensor]]) -> Tuple[LongTensor, LongTensor]:
    x = [x for x, y in batch]
    x_true_length = [len(x) for x, y in batch]
    y = torch.stack([y for x, y in batch], dim=0)

    return ((pad_sequence(x, batch_first=True), LongTensor(x_true_length)), y)

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:24:49 2020
@author: Jiang Yuxin
"""

import random
import copy
import numpy as np

from torch.utils.data import Dataset
import torch


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def split_dataset(dataset, imbalance_factor=10, num_meta_total=200):
    num_classes = 2
    num_meta = int(num_meta_total / num_classes)

    index_to_meta = []
    index_to_train = []

    imbalanced_num_list = []
    sample_num = int((len(dataset.labels) - num_meta_total) / num_classes)
    for class_index in range(num_classes):
        imbalanced_num = sample_num / (
            imbalance_factor ** (class_index / (num_classes - 1))
        )
        imbalanced_num_list.append(int(imbalanced_num))
    print(imbalanced_num_list)

    for class_index in range(num_classes):
        index_to_class = [
            index for index, label in enumerate(dataset.labels) if label == class_index
        ]
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])
        index_to_class_for_train = index_to_class[num_meta:]

        index_to_class_for_train = index_to_class_for_train[
            : imbalanced_num_list[class_index]
        ]

        index_to_train.extend(index_to_class_for_train)

    meta_dataset = copy.deepcopy(dataset)
    dataset.input_ids = dataset.input_ids[index_to_train]
    dataset.attention_mask = dataset.attention_mask[index_to_train]
    dataset.token_type_ids = dataset.token_type_ids[index_to_train]
    dataset.labels = dataset.labels[index_to_train]
    meta_dataset.input_ids = meta_dataset.input_ids[index_to_meta]
    meta_dataset.attention_mask = meta_dataset.attention_mask[index_to_meta]
    meta_dataset.token_type_ids = meta_dataset.token_type_ids[index_to_meta]
    meta_dataset.labels = meta_dataset.labels[index_to_meta]

    return dataset, meta_dataset


class DataPrecessForSentence(Dataset):
    """
    Encoding sentences
    """

    def __init__(self, bert_tokenizer, df, max_seq_len=50):
        super(DataPrecessForSentence, self).__init__()
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_seq_len
        (
            self.input_ids,
            self.attention_mask,
            self.token_type_ids,
            self.labels,
        ) = self.get_input(df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.labels[idx],
        )

    # Convert dataframe to tensor
    def get_input(self, df):
        sentences = df["s1"].values
        labels = df["similarity"].values

        # tokenizer
        tokens_seq = list(
            map(self.bert_tokenizer.tokenize, sentences)
        )  # list of shape [sentence_len, token_len]

        # Get fixed-length sequence and its mask
        result = list(map(self.trunate_and_pad, tokens_seq))

        input_ids = [i[0] for i in result]
        attention_mask = [i[1] for i in result]
        token_type_ids = [i[2] for i in result]

        return (
            torch.Tensor(input_ids).type(torch.long),
            torch.Tensor(attention_mask).type(torch.long),
            torch.Tensor(token_type_ids).type(torch.long),
            torch.Tensor(labels).type(torch.long),
        )

    def trunate_and_pad(self, tokens_seq):

        # Concat '[CLS]' at the beginning
        tokens_seq = ["[CLS]"] + tokens_seq
        # Truncate sequences of which the lengths exceed the max_seq_len
        if len(tokens_seq) > self.max_seq_len:
            tokens_seq = tokens_seq[0 : self.max_seq_len]
        # Generate padding
        padding = [0] * (self.max_seq_len - len(tokens_seq))
        # Convert tokens_seq to token_ids
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens_seq)
        input_ids += padding
        # Create attention_mask
        attention_mask = [1] * len(tokens_seq) + padding
        # Create token_type_ids
        token_type_ids = [0] * (self.max_seq_len)

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len

        return input_ids, attention_mask, token_type_ids

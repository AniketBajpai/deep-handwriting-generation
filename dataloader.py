import os
import numpy as np
import torch
import random

import config
from utils import sentence_to_tensor


class DataLoader(object):
    ''' class to load data during training '''

    def __init__(self):
        self.load_data()

    def load_data(self):
        ''' load strokes and text data from file '''
        self.strokes_train = np.load(
            config.stroke_train_file_path, encoding='latin1')
        self.strokes_test = np.load(
            config.stroke_test_file_path, encoding='latin1')
        with open(config.text_train_file_path) as f:
            self.texts_train = f.readlines()
        with open(config.text_test_file_path) as f:
            self.texts_test = f.readlines()
        assert len(self.strokes_train) == len(self.texts_train)
        assert len(self.strokes_test) == len(self.texts_test)
        self.train_dataset_size = len(self.strokes_train)
        self.test_dataset_size = len(self.strokes_test)

    def get_minibatch(self, task_type, is_conditional):
        ''' generator to yield next minibatch '''
        # Only for batch size 1
        # TODO: add support for general batch sizes
        if task_type == 'gen':
            if is_conditional:
                for idx in range(self.train_dataset_size):
                    stroke = self.strokes_train[idx]
                    text = self.texts_train[idx]
                    stroke_tensor = torch.from_numpy(stroke)
                    y_onehot = sentence_to_tensor(text, config.num_chars)
                    yield (stroke_tensor, y_onehot)
            else:
                for stroke in self.strokes_train:
                    stroke_tensor = torch.from_numpy(stroke)
                    # print(stroke_tensor.size())
                    yield stroke_tensor    # dim (len_stroke, 3)
        elif task_type == 'rec':
            for idx in range(self.train_dataset_size):
                stroke = self.strokes_train[idx]
                text = self.texts_train[idx]
                stroke_tensor = torch.from_numpy(stroke)
                text = config.sos_token + text + config.eos_token
                num_chars = config.num_chars + 3    # <sos>, <eos>, space
                y_onehot = sentence_to_tensor(text, num_chars)
                yield (stroke_tensor, y_onehot)

    def get_random_examples(self, num_examples, phase, task_type, is_conditional):
        if phase == 'train':
            dataset_size = self.train_dataset_size
            strokes = self.strokes_train
            texts = self.texts_train
        elif phase == 'test':
            dataset_size = self.test_dataset_size
            strokes = self.strokes_test
            texts = self.texts_test
        
        if task_type == 'gen':
            num_chars = config.num_chars
        elif task_type == 'rec':
            num_chars = config.num_chars + 3
        
        random_idxs = random.sample(range(dataset_size), num_examples)
        picked_strokes = [torch.from_numpy(strokes[i]) for i in random_idxs]
        if is_conditional:
            picked_texts = [sentence_to_tensor(texts[i], num_chars) for i in random_idxs]

        if is_conditional:
            return zip(picked_strokes, picked_texts)
        else:
            return picked_strokes

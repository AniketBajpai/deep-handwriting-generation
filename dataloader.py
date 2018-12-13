import os
import numpy as np
import torch

import config


class DataLoader(object):
    ''' class to load data during training '''

    def __init__(self):
        self.load_data()

    def load_data(self):
        ''' load strokes and text data from file '''
        self.strokes = np.load(
            config.stroke_train_file_path, encoding='latin1')
        with open(config.text_train_file_path) as f:
            self.texts = f.readlines()

    def sentence_to_tensor(self, sentence):
        chars = list(sentence)
        unique_chars = """!"#'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""
        self.num_chars = config.num_chars
        char_indices = [unique_chars.find(c) for c in chars]
        y_onehot = np.array(char_indices)
        y_onehot = (np.arange(self.num_chars) ==
                    y_onehot[:, None]).astype(np.float32)
        y_onehot = torch.from_numpy(y_onehot)
        return y_onehot

    def get_minibatch(self, is_conditional):
        ''' generator to yield next minibatch '''
        # Only for batch size 1
        # TODO: add support for general batch sizes
        if is_conditional:
            assert len(self.strokes) == len(self.texts)
            l = len(self.texts)
            for idx in range(l):
                stroke = self.strokes[idx]
                text = self.texts[idx]
                stroke_tensor = torch.from_numpy(stroke)
                y_onehot = self.sentence_to_tensor(text)
                yield (stroke_tensor, y_onehot)
        else:
            for stroke in self.strokes:
                stroke_tensor = torch.from_numpy(stroke)
                # print(stroke_tensor.size())
                yield stroke_tensor    # dim (len_stroke, 3)

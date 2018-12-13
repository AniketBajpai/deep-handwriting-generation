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
        self.strokes = np.load(config.stroke_file_path, encoding='latin1')
        with open(config.text_file_path) as f:
            self.texts = f.readlines()
    
    def get_minibatch(self):
        ''' generator to yield next minibatch '''
        # Only for batch size 1
        # TODO: add support for general batch sizes
        for stroke in self.strokes:
            stroke_tensor = torch.from_numpy(stroke)
            # print(stroke_tensor.size())
            yield stroke_tensor    # dim (len_stroke, 3)
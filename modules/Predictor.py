"""
This is supposed to implement section 3.5 of the paper
"""

import torch
import torch.nn as nn
from utils import BiDAFNet


class Predictor(nn.Module):
    #TODO docstring

    def __init__(self):
        super(Predictor, self).__init__() # the following build on this

        self.bidaf = BiDAFNet() # hidden_size=768, output_size=300


    def forward(self, context_emb):
    #TODO dosctring



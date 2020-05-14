"""
This is supposed to implement section 3.5 of the paper
"""

import torch
import torch.nn as nn
from utils import Linear

from utils import make_labeled_data_for_predictor

class Predictor(nn.Module):
    """
    This class implements the Prediction layer from the paper (section 3.5)
    The Weak Supervision part is not included.
    """

    def __init__(self, context_length, embedding_size, dropout=0.0):
        """
        Initialise parameters and layers for Predictor.

        :param context_length: length of the context
        :param embedding_size: hidden embedding size (d2 in the paper)
        :param dropout: dropout rate
        """
        super(Predictor, self).__init__()  # the following build on this

        d2 = embedding_size

        self.f0 = nn.LSTM(d2, d2) # input_site, output_size
        self.f1 = nn.LSTM(2*d2, d2)
        self.f2 = nn.LSTM(3*d2, d2)
        self.f3 = nn.LSTM(3*d2, d2)

        self.linear_sup =   Linear(d2, 2, dropout=dropout)  # have 2 output dims because we need to weight the classes
        self.linear_start = Linear(d2, 1, dropout=dropout)  # with a softmax because there can only be one start or end
        self.linear_end =   Linear(d2, 1, dropout=dropout)
        self.linear_type =  Linear(d2, 3, dropout=dropout)  # 3 because we have 3 types - yes, no, and span


    def forward(self, context_emb):
        """
        Forward function of the Predictor.
        This implements the logic behind the prediction layer from the paper.
        As in the original paper, we also use the structure
        developed by Yang et al. (2018) (https://arxiv.org/pdf/1809.09600.pdf)

        :param context_emb: embedding on the context as produced by the FusionBlock
        :return result: a 4-tuple (( (M), (M), (M), (1,3)), containing:
                            - supporting fact scores (a score for each token)
                            - start scores (a score for each token)
                            - end scores (a score for each token)
                            - answer type score (1,3) - 3 because we have 3 types - yes, no, and span
        """

        Ct = context_emb.unsqueeze(0) # (1, M, d2)

        o_sup, hidden_o_sup = self.f0(Ct)   # (1, M, d_2) -> (1, M, d_2)
        sup_scores = self.linear_sup(o_sup) # (1, M, d_2) -> (1, M, 2) #TODO maybe apply softmax here as well?

        o_start, hidden_o_start = self.f1(torch.cat((Ct, o_sup), dim=-1))  	    # (1, M, 2*d_2) -> (1, M, d_2)
        start_scores = torch.softmax(self.linear_start(o_start), 1) 		    # (1, M, d_2) -> (1, M, 1)

        o_end, hidden_o_end = self.f2(torch.cat((Ct, o_sup, o_start), dim=-1))  # (1, M, 3*d_2) -> (1, M, d_2)
        end_scores = torch.softmax(self.linear_end(o_end), 1) 				    # (1, M, d_2) -> (1, M, 1)

        o_type, hidden_o_type = self.f3(torch.cat((Ct, o_sup, o_end), dim=-1))  # (1, M, 3*d_2) -> (1, M, d_2)
        #o_type =  o_type.view(o_type.shape[1], o_type.shape[0], 1, o_type.shape[2])[-1] # select the last output state: (1, d2) #CLEANUP?
        o_type = o_type.mean(1).squeeze(1) # use mean pooling over the sequence: (1, d2)
        type_scores = self.linear_type(o_type) 		                            # (1, d_2) -> (1, 3)

        result = (sup_scores.squeeze(0), \
                 start_scores.squeeze(-1), \
                 end_scores.squeeze(-1), \
                 type_scores) #TODO? .squeeze(-1) ?

        return result  # ( (M), (M), (M), (1,3) )
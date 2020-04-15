"""
This is supposed to implement section 3.5 of the paper
"""

import torch
import torch.nn as nn
from utils import BiDAFNet


class Predictor(nn.Module):
    #TODO docstring

    def __init__(self, embedding_size):
    	"""
    	TODO: update docstring

    	Initialise parameters and layers for Predictor.

    	:param embedding_size:
        """
        super(Predictor, self).__init__() # the following build on this

        # We think we don't need BiDAF here because the context was
        # already updated by the query (this is what FusionBlock was about)
        # self.bidaf = BiDAFNet() # hidden_size=768, output_size=300

        """
        QUESTIONNNNNSSSSS:

        1. Where does 2*d_2 come from (authors' code, line 365, 368, etc.)
        2. How are the embeddings concatenated for the linear layer (by row or column?)? 
            -> along d2, probably
        3. Why do they use a linear layer for 'support' as well when there's none in the Yang paper
        4. What about self attention?
        5. Where do the coefficients come from? Where are they defined and why? (lambdas in formula 15)
            -> e.g. defined as hyperparameters. Maybe they'r edefined somewhere in their code
        6. Why do they add 1 to input_dim in lines 367, 370 and 373?
        7. What about the weakly supervised signal?
        8. What are heuristic masks?
        9. How do you detect the start mask from the query?
        """

        d2 = embedding_size

        self.f0 = nn.LSTM(d2, d2) # input_site, output_size
        self.f1 = nn.LSTM(2*d2, d2)
        self.f2 = nn.LSTM(3*d2, d2)
        self.f3 = nn.LSTM(3*d2, d2)
        self.linear_start = nn.Linear(d2, 1)
        self.linear_end = nn.Linear(d2, 1)
        self.linear_type = nn.Linear(d2, 3) # 3 because we have 3 types - yes, no, and span


    def forward(self, context_emb):
        """
        TODO dosctring

        :param context_emb: embedding on the context as produced by the FusionBlock
        """

        Ct = context_emb.unsqueeze(0) # (1, M, d2)

    	o_sup, hidden_o_sup = self.f0(Ct) 									   # (1, M, d_2) -> (1, M, d_2)

    	o_start, hidden_o_start = self.f1(torch.cat((Ct, o_sup), dim=-1))  	   # (1, M, 2*d_2) -> (1, M, d_2)
    	start_scores = self.linear_start(o_start) 						       # (1, M, d_2) -> (1, M, 1) # TODO make sure that the batch axis doesn't make trouble

    	o_end, hidden_o_end = self.f2(torch.cat((Ct, o_sup, o_start), dim=-1)) # (1, M, 3*d_2) -> (1, M, d_2)
    	end_scores = self.linear_end(o_end) 								   # (1, M, d_2) -> (1, M, 1)

    	o_type, hidden_o_type = self.f3(torch.cat((Ct, o_sup, o_end), dim=-1)) # (1, M, 3*d_2) -> (1, M, d_2)
    	a_type_scores = self.linear_type(o_type) 							   # (1, M, d_2) -> (1, M, 3) # TODO should this rather be (1, 3)?





    """
    For each of the o_ outputs, we need a tensor of labels in order to compute the loss. 
    This means:
    - o_sup: look at the supporting facts and graph.context: 
        if the paragraph title is in supporting facts, fill graph.tokens with 1s for 
        the corresponding tokens (might need to use a counter)
    - o_type: look at the answers. Each column of the label tensor is one answer type:
        'yes' is column 0, 'no' is column 1, anything else is column 2
    - o_start, o_end: if o_type is 2, then find the start and the end of the span:
        take graph.tokens and look for each token:
        - is it at the beginning of the answer? -> start! (give it a 1 in the start labels)
        - is it at the end of the answer? -> end! (give it a 1 in the end labels)
    """
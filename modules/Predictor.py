"""
This is supposed to implement section 3.5 of the paper
"""

import torch
import torch.nn as nn
from utils import BiDAFNet


class Predictor(nn.Module):
    #TODO docstring

    def __init__(self, input_size, output_size):
    	"""
    	TODO: update docstring

    	Initialise parameters and layers for Predictor.

    	:param input_size:
    	:param output_size:
        """
        super(Predictor, self).__init__() # the following build on this

        # We think we don't need BiDAF here because the context was
        # already updated by the query (this is what FusionBlock was about)
        # self.bidaf = BiDAFNet() # hidden_size=768, output_size=300

        """
        QUESTIONNNNNSSSSS:

        1. Where does 2*d_2 come from (authors' code, line 365, 368, etc.)
        2. How are the embeddings concatenated for the linear layer (by row or column?)?
        3. Why do they use a linear layer for 'support' as well when there's none in the Yang paper
        4. What about self attention?
        5. Where do the coefficients come from? Where are they defined and why? (lambdas in formula 15)
        6. Why do they add 1 to input_dim in lines 367, 370 and 373?
        7. What about the weakly supervised signal?
        8. What are heuristic masks?
        9. How do you detect the start mask from the query?
        """

        self.f0 = nn.LSTM(,)
        self.f1 = nn.LSTM(,)
        self.f2 = nn.LSTM(,)
        self.f3 = nn.LSTM(,)
        self.linear_start = nn.Linear(, 1)
        self.linear_end = nn.Linear(, 1)
        self.linear_type = nn.Linear(, 3) # 3 because we have 3 types - yes, no, and span


    def forward(self, context_emb):
    """
    TODO dosctring
    
    :param context_emb: embedding on the context as produced by the FusionBlock
    """
    	o_sup = self.f0(context_emb) 									# (M, d_2) -> (M, d_2)

    	o_start = self.f1(torch.cat((context_emb, o_sup), dim=-1))  	# (M, 2*d_2) -> (M, d_2)
    	start_token = self.linear_start(o_start) 						# (M, d_2) -> 1

    	o_end = self.f2(torch.cat((context_emb, o_sup, o_start), dim=-1)) # (M, 3*d_2) -> (M, d_2)
    	end_token = self.linear_end(o_end) 								# (M, d_2) -> 1

    	o_type = self.f3(torch.cat((context_emb, o_sup, o_end), dim=-1))# (M, 3*d_2) -> (M, d_2)
    	a_type = self.linear_type(o_type) 								# (M, d_2) -> 3



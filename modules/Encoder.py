"""
This module implements the Encoder from the paper (Section 3.3)
"""

import torch
from transformers import BertTokenizer, BertModel
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from utils import flatten_context, Linear, BiDAFNet
import torch.nn as nn
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    """
    #TODO update dosctring
    Use BERT and bidirectional attention flow (BiDAF) to token_ids a query and a
    context. Both BERT and the BiDAF component are trained.
    """
    # relicts from the EncoderBiDAF class:
    """
    This class implements bidirectional attention flow (BiDAF) as
    described in Seo et al. (2016): arxiv.org/pdf/1611.01603.pdf
    The subsequent code is copied in most parts from Taeuk Kim's
    Pytorch re-implementation of the above paper:
    https://github.com/galsang/BiDAF-pytorch
    """

    def __init__(self, text_length=512, pad_token_id=0, tokenizer=None,
                 hidden_size=768, output_size=300, encoder_model=None):
        """
        Instantiate a Bert tokenizer and a BiDAF net which contains the BERT encoder.
        Sizes of input and output (768,300) are not implemented to be changeable.
        :param text_length: maximum number of tokens (query+context)
        :param pad_token_id: for padding to text_length
        :param tokenizer: defaults to 'bert-base-uncased'
        :param encoder_model: defaults to 'bert-base-uncased'
        """
        super(Encoder, self).__init__()

        self.text_length = text_length
        self.pad_token_id = pad_token_id

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') if not tokenizer else tokenizer
        self.encoder_model = BertModel.from_pretrained('bert-base-uncased',
                         output_hidden_states=True, # TODO leave out output_attentions? (This implies some other changes!)
                         output_attentions=True) if not encoder_model else encoder_model
        self.bidaf = BiDAFNet(hidden_size=hidden_size, output_size=output_size)

    def forward(self, q_token_ids, c_token_ids, batch=1):
        """
        Encode a query and a context (both lists of token IDs) and
        apply BiDAF to the encoding.
        :param q_token_ids: list[ine] or Tensor[int] - obtained from a tokenizer
        :param c_token_ids: list[int] or Tensor[int] - obtained from a tokenizer
        :return: encoded and BiDAF-ed context of shape (batch, c_len, output_size)
        """

        #TODO maybe change this in order to avoid unnecessary computing?
        if type(q_token_ids) == torch.Tensor:
            q_token_ids = q_token_ids.tolist()
        if type(c_token_ids) == torch.Tensor:
            c_token_ids = c_token_ids.tolist()



        all_token_ids = q_token_ids + c_token_ids
        len_all = len(all_token_ids)
        len_query = len(q_token_ids)
        len_context = len(c_token_ids)

        #TODO 23.04.2020: identify the longer and the shorter one out of c and q and then handle them

        # Add padding or trim to text_length
        if len_all < self.text_length:
            all_token_ids += [self.pad_token_id for _ in range(self.text_length - len_all)]
        else:
            if len_context >= len_query: # make sure that the longer one will be trimmed!
                trim_this = c_token_ids
                attach_this = q_token_ids
            else:
                trim_this = q_token_ids
                attach_this = c_token_ids
            all_token_ids = trim_this[:len_all-len(attach_this)+1] + attach_this



        # get the embeddings corresponding to the token IDs
        all_hidden_states, all_attentions = self.encoder_model(torch.tensor([all_token_ids]))[-2:]

        # Next five lines:
        # This is the embedding of the context + query
        # [-1] = last hidden state
        # [0] = first sentence ('sentence' = sequence of characters)
        q_emb = all_hidden_states[-1][0][:len_query]

        # If query + context is longer than text_length (512 by default),
        # the context embedding includes everything except the query
        if len_all > self.text_length:
            c_emb = all_hidden_states[-1][0][len_query:]
        # Else (query + context shorter than text_length),
        # the context embedding will start after the query embedding,
        # and end after len_query+len_context elements
        # This will prevent us from taking the padding embeddings as
        # parts of the context embedding
        else:
            c_emb = all_hidden_states[-1][0][len_query:len_query+len_context]

        #print("Query shape:", q_emb.shape) #CLEANUP?
        #print("Context shape:", c_emb.shape)
        #print(q_emb)
        #print(c_emb)

        g = self.bidaf(q_emb, c_emb)
        return g


    def token_ids(self, query=None, context=None):
        """
        return the token IDs of a query and a context
        :param query: str
        :param context: setences, paragraphs, paragraph titles
        :type context: list[list[str,list[str]]]
        :return: list[int], list[int] -- query token IDs, context token IDs
        """
        if not query:
            print("No query for Encoder. Working with toy example.")
            query = "Who had a little lamb?"

        if not context:
            print("No context for Encoder. Working with toy example.")
            context = [
                ["Mary and her lamb",
                 ["Mary had a little lamb.",
                  " The lamb was called Tony.",
                  " One day, Bill Gates wanted to hire Tony."]],
                ["All like it but John",
                 ["Siyana thought that Tony is cute.",
                  " Well, I also think that he is nice.",
                  " Mary, however liked Tony even more than we do."]]
            ]


        # Tokenize and token_ids the query and the context
        query_input_ids = self.tokenizer.encode(query,
                                                   add_special_tokens=False,
                                                   max_length=self.text_length)
        context_input_ids = self.tokenizer.encode(flatten_context(context),
                                                     add_special_tokens=False,
                                                     max_length=self.text_length)

        return query_input_ids, context_input_ids


    #def predict(self, q_token_ids, c_token_ids): #CLEANUP?
    #    return self.bidaf(q_token_ids, c_token_ids)



    
    
    

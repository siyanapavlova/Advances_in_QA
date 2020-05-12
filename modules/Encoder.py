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
                 hidden_size=768, output_size=300, dropout=0.0, encoder_model=None):
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
        self.bidaf = BiDAFNet(hidden_size=hidden_size,
                              output_size=output_size,
                              dropout=dropout)

    def forward(self, q_token_ids, c_token_ids, batch=1):
        """
        Encode a query and a context (both lists of token IDs) and
        apply BiDAF to the encoding.
        :param q_token_ids: list[ine] or Tensor[int] - obtained from a tokenizer
        :param c_token_ids: list[int] or Tensor[int] - obtained from a tokenizer
        :return: encoded and BiDAF-ed context of shape (batch, c_len, output_size)
        """
        MAX_LEN = 512

        #TODO rename variables to avoid confusion!!!

        #TODO maybe change this in order to avoid unnecessary computing?

        len_query = q_token_ids.shape[0]
        len_context = c_token_ids.shape[0]

        # we need to trim, otherwise (1) Bert will explode or (2) our context is longer than specified
        if (len_query + len_context > MAX_LEN) or (len_query + len_context > self.text_length):
            cut_point = min(MAX_LEN - len_query, self.text_length)
            if len_context >= len_query:  # trim whatever 'context' is
                c_token_ids = c_token_ids[:cut_point]
                len_context = c_token_ids.shape[0]
            else:  # trim whatever 'query' is
                q_token_ids = q_token_ids[:cut_point]
                len_query = q_token_ids.shape[0]

        all_token_ids = torch.cat((q_token_ids, c_token_ids))
        len_all = all_token_ids.shape[0]

        # we need to pad
        if len_query + len_context < self.text_length:
            padding = torch.tensor([self.pad_token_id for _ in range(self.text_length - len_all)],
                                   device=all_token_ids.device,
                                   dtype=all_token_ids.dtype)
            all_token_ids = torch.cat((all_token_ids, padding))


        # get the embeddings corresponding to the token IDs
        all_hidden_states, all_attentions = self.encoder_model(all_token_ids.unsqueeze(0))[-2:]

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

        # TODO check whether we actually always return something with text_length!!!
        #  (in cases with large text_length and >512, we might return a context that is shorter than text_length)
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

        # Add padding if there are fewer than text_length tokens,
        if len(context_input_ids) < self.text_length:
            context_input_ids += [self.tokenizer.pad_token_id
                                  for _ in
                                  range(self.text_length - len(context_input_ids))]

        return query_input_ids, context_input_ids


    #def predict(self, q_token_ids, c_token_ids): #CLEANUP?
    #    return self.bidaf(q_token_ids, c_token_ids)



    
    
    

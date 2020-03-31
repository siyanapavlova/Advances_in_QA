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

class Encoder():
    """
    Use BERT and bidirectional attention flow (BiDAF) to encode a query and a
    context. Both BERT and the BiDAF component are trained.
    """

    def __init__(self, text_length=512, pad_token_id=0, tokenizer=None, encoder_model=None):
        """
        Instantiate a Bert tokenizer and a BiDAF net which contains the BERT encoder.
        Sizes of input and output (768,300) are not implemented to be changeable.
        :param text_length: maximum number of tokens (query+context)
        :param pad_token_id: for padding to text_length
        :param tokenizer: defaults to 'bert-base-uncased'
        :param encoder_model: defaults to 'bert-base-uncased'
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') if not tokenizer else tokenizer
        
        class EncoderBiDAF(torch.nn.Module):
            """
            This class implements bidirectional attention flow (BiDAF) as
            described in Seo et al. (2016): arxiv.org/pdf/1611.01603.pdf
            The subsequent code is copied in most parts from Taeuk Kim's
            Pytorch re-implementation of the above paper:
            https://github.com/galsang/BiDAF-pytorch
            """
            def __init__(self, hidden_size=768, output_size=300, encoder_model=None):
                """
                Make an encoder and attention layers as well as an output layer.
                :param hidden_size: input (usually from a BERT encoder)
                :param output_size: output (specified to 300)
                :param encoder_model: defaults to 'bert-base-uncased'
                """
                super(EncoderBiDAF, self).__init__()

                self.encoder_model = BertModel.from_pretrained('bert-base-uncased',
                                 output_hidden_states=True,
                                 output_attentions=True) if not encoder_model else encoder_model

                self.bidaf = BiDAFNet(hidden_size=hidden_size, output_size=output_size)

            def forward(self, q_token_ids, c_token_ids, batch=1):
                """
                Encode a query and a context (both lists of token IDs) and
                apply BiDAF to the encoding.
                :param q_token_ids: list[int] - obtained from a tokenizer
                :param c_token_ids: list[int] - obtained from a tokenizer
                :return: encoded and BiDAF-ed context of shape (batch, c_len, output_size)
                """
                
                len_query = len(q_token_ids)
                len_context = len(c_token_ids)
                
                all_token_ids = q_token_ids + c_token_ids

                # Add padding or trim to text_length
                if len(all_token_ids) < text_length:
                    all_token_ids += [pad_token_id for _ in range(text_length - len(all_token_ids))]
                else:
                    all_token_ids = all_token_ids[:text_length]                

                # get the embeddings corresponding to the token IDs
                all_hidden_states, all_attentions = self.encoder_model(torch.tensor([all_token_ids]))[-2:]
                
                # Next five lines: 
                # This is the embedding of the context + query
                # [-1] = last hidden state
                # [0] = first sentence ('sentence' = sequence of characters)
                q_emb = all_hidden_states[-1][0][:len_query]
                
                # If query + context is longer than text_length (512 by default),
                # the context embedding includes everything except the query
                if len(all_token_ids) > text_length:
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
        
        self.net = EncoderBiDAF(encoder_model=encoder_model)

    def encode(self, query=None, context=None):
        """
        return the token IDs of a query and a context
        :param query: str
        :param context: setences, paragraphs, paragraph titles
        :type context: list[list[str,list[str]]]
        :return:
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


        # Tokenize and encode the query and the context
        query_input_ids = self.tokenizer.encode(query,
                                                add_special_tokens=False,
                                                max_length=512)
        context_input_ids = self.tokenizer.encode(flatten_context(context),
                                                  add_special_tokens=False,
                                                  max_length=512)

        return query_input_ids, context_input_ids
    
    def train(self, q_token_ids, c_token_ids):
        self.net.forward(q_token_ids, c_token_ids)
        

if __name__=="__main__":
    e = Encoder()
    q_ids, c_ids = e.encode()
    e.train(q_ids, c_ids)
    
    
    
    

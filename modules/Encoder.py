"""
This module implements the Encoder from the paper (Section 3.3)
"""

import torch
from transformers import BertTokenizer, BertModel
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from utils import flatten_context, Linear
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
        
        class BiDAFNet(torch.nn.Module):
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
                super(BiDAFNet, self).__init__()

                self.encoder_model = BertModel.from_pretrained('bert-base-uncased',
                                 output_hidden_states=True,
                                 output_attentions=True) if not encoder_model else encoder_model
                
                self.att_weight_c = Linear(hidden_size, 1)
                self.att_weight_q = Linear(hidden_size, 1)
                self.att_weight_cq = Linear(hidden_size, 1)

                self.reduction_layer = Linear(hidden_size * 4, output_size)

            def forward(self, q_token_ids, c_token_ids, batch=1):
                """
                Encode a query and a context (both lists of token IDs) and
                apply BiDAF to the encoding.
                :param q_token_ids: list[int] - obtained from a tokenizer
                :param c_token_ids: list[int] - obtained from a tokenizer
                :return: encoded and BiDAF-ed context of shape (batch, c_len, output_size)
                """

                def att_flow_layer(q, c):
                    """
                    Perform BiDAF and return the updated context.
                    :param q: encoded query of shape (batch, q_len, hidden_size)
                    :param c: encoded context of shape (batch, c_len, hidden_size)
                    :return: encoded context (batch, c_len, output_size)
                    """
                    c_len = c.size(1)
                    q_len = q.size(1)

                    cq = []
                    for i in range(q_len):
                        qi = q.select(1, i).unsqueeze(1) # (batch, 1, hidden_size)
                        ci = self.att_weight_cq(c * qi).squeeze(-1) # (batch, c_len, 1)
                        cq.append(ci)
                    cq = torch.stack(cq, dim=-1) # (batch, c_len, q_len)

                    # (batch, c_len, q_len)
                    s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                        self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                        cq

                    a = F.softmax(s, dim=2) # (batch, c_len, q_len)
                    # (batch, c_len, q_len) * (batch, q_len, hidden_size) -> (batch, c_len, hidden_size)
                    c2q_att = torch.bmm(a, q)

                    b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)  # (batch, 1, c_len)
                    # (batch, 1, c_len) * (batch, c_len, hidden_size) -> (batch, hidden_size)
                    q2c_att = torch.bmm(b, c).squeeze(1)
                    # (batch, c_len, hidden_size) (tiled)
                    q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)

                    # (batch, c_len, hidden_size * 4)
                    x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
                    x = self.reduction_layer(x) # (batch, c_len, output_size)
                    return x
                
                
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
            
                g = att_flow_layer(q_emb, c_emb)
                return g
        
        self.net = BiDAFNet(encoder_model=encoder_model)

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
    
    
    
    

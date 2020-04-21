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
    TODO: write docstring
    """

    def __init__(self,
                 tokenizer=None,
                 encoder_model=None):
        """
        TODO: write docstring
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') if not tokenizer else tokenizer
        self.encoder_model = BertModel.from_pretrained('bert-base-uncased',
                                 output_hidden_states=True,
                                 output_attentions=True) if not encoder_model else encoder_model
        
        class BiDAFNet(torch.nn.Module):
            """
            TODO: write docstring

            BiDAF paper: arxiv.org/pdf/1611.01603.pdf
            There's a link to the code, but that uses TensorFlow

            We adapted this implementation of the BiDAF
            Attention Layer: https://github.com/galsang/BiDAF-pytorch
            """
            def __init__(self, hidden_size=768, output_size=300):
                super(BiDAF, self).__init__()
                
                self.att_weight_c = Linear(hidden_size, 1)
                self.att_weight_q = Linear(hidden_size, 1)
                self.att_weight_cq = Linear(hidden_size, 1)

                self.reduction_layer = Linear(hidden_size * 4, output_size)

            def forward(self, c, q, batch=1):

                def att_flow_layer(c, q):
                    """
                    :param c: (batch, c_len, hidden_size * 2)
                    :param q: (batch, q_len, hidden_size * 2)
                    :return: (batch, c_len, q_len)
                    """
                    c_len = c.size(1)
                    q_len = q.size(1)

                    cq = []
                    for i in range(q_len):
                        #(batch, 1, hidden_size * 2)
                        qi = q.select(1, i).unsqueeze(1)
                        #(batch, c_len, 1)
                        ci = self.att_weight_cq(c * qi).squeeze(-1)
                        cq.append(ci)
                    # (batch, c_len, q_len)
                    cq = torch.stack(cq, dim=-1)

                    # (batch, c_len, q_len)
                    s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                        self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                        cq

                    # (batch, c_len, q_len)
                    a = F.softmax(s, dim=2)

                    # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
                    c2q_att = torch.bmm(a, q)

                    # (batch, 1, c_len)
                    b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)

                    # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
                    q2c_att = torch.bmm(b, c).squeeze(1)

                    # (batch, c_len, hidden_size * 2) (tiled)
                    q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)

                    # (batch, c_len, hidden_size * 8)
                    x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)

                    x = self.reduction_layer(x)
                    return x

                g = att_flow_layer(c, q)
                return g

    def encode(self,
               query=None,
               context=None):
        """
        TODO: write docstring
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
            
        concatenated = query + ' ' + flatten_context(context)
        
        tokenized_query = self.tokenizer.tokenize(query)
        tokenized_context = self.tokenizer.tokenize(flatten_context(context))
        
        len_query = len(tokenized_query)
        
        input_ids = torch.tensor([self.tokenizer.token_ids(concatenated, add_special_tokens=False)])
        all_hidden_states, all_attentions = self.encoder_model(input_ids)[-2:]

        # This is the embedding of the context + query
        # [-1] stands for the last hidden state
        # sentence being defined as a sequence of characters, and not a linguistic sentence)
        return all_hidden_states[-1][0][:len_query], all_hidden_states[-1][0][len_query:]

if __name__=="__main__":
    e = Encoder()
    query_emb, context_emb = e.encode()
    
    q_emb_unsqueezed = q_emb.unsqueeze(0)
    c_emb_unsqueezed = c_emb.unsqueeze(0)
    
    print("Query shape:", query_emb.shape)
    print("Context shape:", context_emb.shape)
    print(query_emb)
    print(context_emb)
    

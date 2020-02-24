"""
This module implements the Encoder from the paper (Section 3.3)
"""

import torch
from transformers import BertTokenizer, BertModel
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from utils import flatten_context

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
            """
            def __init__(self, input_size=768, output_size=300):
                pass

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
                     " Some Microsoft executives wanted to hire Tony."]],
                ["All like it but John",
                    ["Siyana thought that Tony is cute.",
                     " Well, I also think that he is nice.",
                     " Mary, however liked Tony even more than we do."]]
            ]
            
        concatenated = query + ' ' + flatten_context(context)
        
        tokenized_query = self.tokenizer.tokenize(query)
        tokenized_context = self.tokenizer.tokenize(flatten_context(context))
        
        len_query = len(tokenized_query)
        
        input_ids = torch.tensor([self.tokenizer.encode(concatenated, add_special_tokens=False)])
        all_hidden_states, all_attentions = self.encoder_model(input_ids)[-2:]

        # This is the embedding of the context + query
        # [-1] stands for the last hidden state
        # sentence being defined as a sequence of characters, and not a linguistic sentence)
        return all_hidden_states[-1][0][:len_query], all_hidden_states[-1][0][len_query:]

if __name__=="__main__":
    e = Encoder()
    query_emb, context_emb = e.encode()
    print("Query shape:", query_emb.shape)
    print("Context shape:", context_emb.shape)
    print(query_emb)
    print(context_emb)
    

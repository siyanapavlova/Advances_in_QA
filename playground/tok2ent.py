"""
This is a development script for the tok2ent step in the Confusion block.
Once the function is working, it will be implemented as method of EntityGraph.
"""

from transformers import BertTokenizer
from modules.EntityGraph import EntityGraph

def tok2ent(graph, context):
    #TODO docstring
    """

    :return:
    """
    """
    How to convert sentence-wide character spans into context-wide spans?
    - go through graph, ID-wise (make sure the it's sorted by IDs), add 
      previous sentences' combined length (PLUS SPACE)
    - add the previous'sentence's length to the combined length if the current
        sentence number is different from the one before.

    How to map character spans onto token positions?
    - use BERT Tokenizer to get a list of tokens 
    - get rid of hash tags, insert spaces, ...
    - go through the one-sentence context by moving in steps of token length
      (use the tokens from BERT Tokenizer)
    - if we arrive at an index close to a 
           - entity start position, we regex-search for the first mention
                in the remaining context, we consume the next BERT token and
                map its list index to the entity (and its span)
                (still doing token consumption and position change)
           - entity end position, we include that token's list index into
                the mapping-thingy and set the position to/after entity's 
                span end. 
    - this gives a dic{entity_ID:[tokenIndices]}


    """
    node_IDs = sorted(graph.keys()) # make sure that the IDs are sorted
    abs_positions = {} # { abs_start : (node_ID, abs_end) }




    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    one_string_context = self.flatten_context()
    tokens = tokenizer.tokenize(one_string_context)



    pass


#==============
g = EntityGraph() # in the end applications, this will be initialized with a context

tok2ent(g.graph, g.context)
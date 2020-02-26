"""

"""

import numpy #TODO?

import torch

def doc2graph(encoded_context, graph):
    """
    #TODO docstring
    :param encoded_context: Tensor of shape (context_len, d2)
    :param graph: EntityGraph (graph.M is a numpy array, shape (context_len, #ent) )
    :return:
    """

    # make a tensor only containing the entities (shape: (d2, #ent))
    entityT = torch.matmul(encoded_context.T, torch.Tensor(graph.M))
    torch.avg_pool
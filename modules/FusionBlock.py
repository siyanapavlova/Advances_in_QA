"""
This implements the Fusion block from the paper (Section 3.4)
"""
from math import sqrt, exp
import torch
import torch.nn.functional as F
import numpy as np

from utils import Linear, BiDAFNet

class FusionBlock():
	"""
	"""

	def __init__(self, context_emb, query_emb, graph):
		"""
		#TODO docstring
		:param context_emb:
		:param query_emb:
		:param graph:
		"""
		self.context_emb = context_emb # M x d_2
		self.query_emb = query_emb # L x d_2
		self.bin_M = graph.M # M x N
		self.graph = graph # EntityGraph object

		self.entity_embs = None # input entity embeddings

		self.bidaf = BiDAFNet()


	def forward(self):
		#TODO docstring
		self.entity_embs = self.tok2ent()
		updated_entity_embs = self.graph_attention()

		self.query_emb = self.bidaf(updated_entity_embs, self.query_emb) # update query


	def tok2ent(self):
		"""
		Document to Graph Flow from the paper (section 3.4)

		:param context_emb: M x d_2 (context embedding from Encoder)
		:param bin_M: M x N (binary matrix from EntityGraph)
		:return : (2d_2, N)
		"""
		M = self.context_emb.shape[0]
		N = self.bin_M.shape[1]

		entity_emb = self.context_emb.expand(-1, N, -1) # M x N x d_2
		bin_M_prime = self.bin_M.unsqueeze(2) # M x N x 1
		entity_emb = entity_emb * bin_M_prime # M x N x d_2
		entity_emb = entity_emb.permute(1, 2, 0) # M x N x d_2 -> N x d_2 x M

		# For the next two lines: (N, d_2, M) -> (N, d_2, 1) -> (N, d_2)
 		mean_pooling = F.avg_pool1d(entity_emb, kernel_size=M).squeeze(-1)
		max_pooling = F.max_pool1d(entity_emb, kernel_size=M).squeeze(-1)

		entity_emb = torch.cat((mean_pooling, max_pooling), dim=-1) # (N, 2d_2)

		return entity_emb.T # (2d_2, N)

	def graph_attention(self):
		"""
		#TODO docstring
		:return:
		"""
		N = self.e_embs.shape[1] # number of entities, taken from  (2d_d, N)
		assert N == len(self.graph) # CLEANUP? # N should be equal to the number of graph nodes

		# formula 1 # (L, d_2) --> (1, L, d_2) --> (1, d_2, L) --> (1, d_2, 1)
		q_emb = F.avg_pool1d(self.query_emb.unsqueeze(0).permute(0, 2, 1),
							 kernel_size=self.query_emb.shape[0])
		q_emb = q_emb.permute(0, 2, 1).squeeze(0) # (1, 1, d_2) --> (1, d_2)

		d_2 = q_emb.shape[1]
		root = sqrt(d_2) # for formula 2
		V = torch.Tensor(d_2, 2*d_2) # (d_2, 2d_2) #TODO is this learned? is it random? find out!

		# N * ( (1, d_2) x (d_2, 2_d2) x (2d_2, 1) ) --> (N, 1, 1) # formula 2
		gammas = torch.tensor([ torch.chain_matmul(q_emb, V, e.T)/root for e in self.e_embs.T ])
		mask = torch.sigmoid(gammas)   # (N, 1, 1) # formula 3
		E = torch.tensor([m*e for m,e in zip(mask, self.e_embs.T)])  # (N, 1, 2d_2) # formula 4
		E = E.squeeze(1) # (N, 2d_2) #TODO do we really need to squeeze?


		""" disseminate information across the dynamic sub-graph """
		U = torch.Tensor(d_2, 2*d_2) # # for formula 5; 'squishes' entities into d_2 again
		bias = torch.Tensor(d_2, 1) #TODO or (1, 2d_2), or just a scalar?
		W = torch.Tensor(2*d_2, 1) # for formula 6
		betas = torch.zeros(N, N)
		alphas = torch.zeros(N, N) # scores of how much information flows from i to the j

		# (d_2, 2d_2) * (1, 2d_2) --> (N, d_2, 2d_2) # formula 5
		hidden = torch.tensor([U * e + bias for e in E]) # TODO is this elementwise or matmul?

		for i, h_i in enumerate(hidden): # h_i.shape = (d_2, 2d_2)
			for j, rel_type in self.graph[i]["links"]: # only for neighbor nodes
				pair = torch.cat((h_i, hidden[j])) # 2d_2 x 2d_2 # TODO CONTINUE HERE: read the GAT paper, ...
				betas[i][j] = F.leaky_relu(torch.matmul(W.T, pair)) # formula 6

			sumex = sum([exp(betas[i][j]) for j in range(N)]) # TODO how to handle cases of betas[i][j]==0?
			for j in range(N): # compute scores for all node combinations
				alphas[i][j] =  exp(betas[i][j]) / sumex # formula 7

		""" compute total information received per node """
		E_t = [] # N * d_2 x 1

		for i in range(N):
			score_sum = sum([alphas[j][i] * hidden[j] for j, rel_type in self.graph[i]["links"]])
			E_t.append(F.relu(score_sum)) # formula 8

		return torch.Tensor(E_t) #TODO these have to be of shape (2d_d, N)


	def graph2doc(self, e_embs):
		"""
		#TODO docstring
		:return:
		"""









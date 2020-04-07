"""
This implements the Fusion block from the paper (Section 3.4)
"""
from math import sqrt, exp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import Linear, BiDAFNet

class FusionBlock(nn.Module):
	"""
	"""

	def __init__(self, context_emb, query_emb, graph):
		"""
		#TODO docstring
		:param context_emb:
		:param query_emb:
		:param graph:
		"""
		super(FusionBlock, self).__init__()

		self.context_emb = context_emb # M x d_2
		self.query_emb = query_emb # L x d_2
		self.bin_M = graph.M # M x N
		self.graph = graph # EntityGraph object

		d2 = self.query_emb.shape[1]
		self.droot = sqrt(d2) 						  # for formula 2
		self.V = nn.Parameter(torch.Tensor(d2, 2*d2)) # for formula 2
		self.U = nn.Parameter(torch.Tensor(d2, 2*d2)) # for formula 5
		self.b = nn.Parameter(torch.Tensor(d2))       # for formula 5
		self.W = nn.Parameter(torch.Tensor(2*d2, 1))  # for formula 6

		self.bidaf = BiDAFNet()



	def forward(self, passes=1):
		#TODO docstring
		#TODO impolement the multiple passes here
		self.entity_embs = self.tok2ent() # (2d_2, N)
		updated_entity_embs = self.graph_attention() # (N, d2)

		# the second one is updated; that's why it's the other way round as in the DFGN paper
		self.query_emb = self.bidaf(updated_entity_embs, self.query_emb) # formula 9


	def tok2ent(self):
		"""
		Document to Graph Flow from the paper (section 3.4)
		#TODO update docstring
		:param context_emb: (batch, M, d2) (context embedding from Encoder)
		:param bin_M: (M, N) (binary matrix from EntityGraph)
		:return : (batch, 2d2, N)
		"""
		M = self.context_emb.shape[1]
		N = self.bin_M.shape[1]
		#print(f"context_emb: {self.context_emb.shape}")#CLEANUP
		#print(f"bin_M: {self.bin_M.shape}")  # CLEANUP

		entity_emb = self.context_emb.unsqueeze(2).expand(-1, -1, N, -1) # (batch, M, N, d2)
		#print(f"entity_emb1: {entity_emb.shape}")  # CLEANUP

		bin_M_prime = self.bin_M.unsqueeze(2) # (M, N, 1)
		#print(f"bin_M_prime: {bin_M_prime.shape}")  # CLEANUP

		entity_emb = entity_emb * bin_M_prime # (batch, M, N, d2) * (M, N 1) = (batch, M, N, d2)
		#print(f"entity_emb2: {entity_emb.shape}")  # CLEANUP

		entity_emb = entity_emb.permute(0, 2, 3, 1) # (batch, M, N, d2) -> (batch, N, d2, M)
		#print(f"entity_emb3: {entity_emb.shape}")  # CLEANUP

		# For the next lines: (batch, N, d2, M) -> (batch, N, d2, 1) -> (batch, N, d2)
		means, maxes = [], []
		for point in entity_emb: #TODO try to avoid the for loop (maybe by batch pooling?)
			means.append(F.avg_pool1d(point, kernel_size=M).squeeze(-1))
			maxes.append(F.max_pool1d(point, kernel_size=M).squeeze(-1))

		mean_pooling = torch.stack(means)
		max_pooling = torch.stack(maxes)

		#mean_pooling = F.avg_pool1d(entity_emb, kernel_size=M).squeeze(-1)
		#max_pooling = F.max_pool1d(entity_emb, kernel_size=M).squeeze(-1)

		entity_emb = torch.cat((mean_pooling, max_pooling), dim=-1) # (batch, N, 2d2)
		#print(f"entity_emb4: {entity_emb.shape}")  # CLEANUP

		return entity_emb.permute(0, 2, 1) # (batch, 2d2, N)

	def graph_attention(self):
		"""
		#TODO docstring
		:return:
		"""
		#TODO avoid for-loops, but first make the method run.
		#TODO avoid torch.Tensor where possible.
		#TODO change all this to comply with batches! But before, think about the structure of this whole module.
		N = self.entity_embs.shape[2] # number of entities, taken from  (batch, 2d2, N) # already altered to comply with batches
		assert N == len(self.graph.graph) # CLEANUP? # N should be equal to the number of graph nodes

		# formula 1 # (L, d2) --> (1, L, d2) --> (1, d2, L) --> (1, d2, 1)
		q_emb = F.avg_pool1d(self.query_emb.unsqueeze(0).permute(0, 2, 1),
							 kernel_size=self.query_emb.shape[0])
		q_emb = q_emb.permute(0, 2, 1).squeeze(0) # (1, 1, d2) --> (1, d2)

		# N * ( (1, d2) x (d2, 2d2) x (2d2, 1) ) --> (N, 1, 1) # formula 2
		gammas = torch.tensor([ torch.chain_matmul(q_emb, self.V, e.T)/self.droot for e in self.entity_embs.T ]) #TODO avoid for-loop and torch.tensor()
		mask = torch.sigmoid(gammas)   # (N, 1, 1) # formula 3
		E = torch.tensor([m*e for m,e in zip(mask, self.entity_embs.T)])  # (N, 1, 2d2) # formula 4
		E = E.squeeze(1).T # (N, 2d2) --> (2d2, N) #TODO do we really need to squeeze?


		""" disseminate information across the dynamic sub-graph """
		betas = torch.zeros(N, N)
		alphas = torch.zeros(N, N) # scores of how much information flows from i to the j

		# N times [(d2, 2d2) * (2d2, 1)] --> (N, d2, 1)  # formula 5
		hidden = torch.tensor([torch.matmul(self.U,e) + self.b for e in E]) # TODO avoid the for-loop

		for i, h_i in enumerate(hidden): # h_i.shape = (d2, 1) #TODO try to avoid these for-loops
			for j, rel_type in self.graph[i]["links"]: # only for neighbor nodes
				pair = torch.cat((h_i, hidden[j])) # (2d2, 1)
				betas[i][j] = F.leaky_relu(torch.matmul(self.W.T, pair)) # formula 6

			sumex = sum([exp(betas[i][j]) for j in range(N)]) # TODO how to handle cases of betas[i][j]==0?
			for j in range(N): # compute scores for all node combinations
				alphas[i][j] =  exp(betas[i][j]) / sumex # formula 7

		""" compute total information received per node """
		E_t = [] #really N * (d2, 1)?

		for i in range(N):
			# scalar * (j, d2, 1) --> sum --> (d2, 1)
			score_sum = sum([alphas[j][i] * hidden[j] for j, rel_type in self.graph[i]["links"]])
			# --> relu --> (d2, 1)
			E_t.append(F.relu(score_sum)) # formula 8

		return torch.Tensor(E_t).squeeze(dim=-1).T # (N, d2) #TODO avoid torch.Tensor()


	def graph2doc(self, entity_embs):
		"""
		#TODO docstring
		:return:
		"""









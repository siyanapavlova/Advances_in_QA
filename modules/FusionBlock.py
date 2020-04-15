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
	This class implements the components of the Fusion Block
	from the paper (Dodument to Graph Flow, Dynamic Graph Attention
	and Graph to Document Flow; Updating Query is done using the BiDAF
	class defined in utils), and puts them together in its forward function.
	"""

	def __init__(self, context_emb, query_emb, graph):
		"""
		Initialization function for the FusionBlock class

		:param context_emb: context embedding as obtained from Encoder, (M, d2)
		:param query_emb: query embdding as obrained from Encoder, (L, d2)
		:param graph: an entity graph as obrained from EntityGraph
		"""
		super(FusionBlock, self).__init__()

		self.context_emb = context_emb # (M, d2) (context embedding from Encoder)
		self.query_emb = query_emb # (L, d2)
		self.bin_M = graph.M # (M, N) (binary matrix from EntityGraph)
		self.graph = graph # EntityGraph object

		d2 = self.query_emb.shape[1]
		self.droot = sqrt(d2) 						  # for formula 2
		self.V = nn.Parameter(torch.Tensor(d2, 2*d2)) # for formula 2
		self.U = nn.Parameter(torch.Tensor(d2, 2*d2)) # for formula 5
		self.b = nn.Parameter(torch.Tensor(d2, 1))    # for formula 5
		self.W = nn.Parameter(torch.Tensor(2*d2, 1))  # for formula 6

		self.bidaf = BiDAFNet(hidden_size=300)

		self.g2d_layer = nn.LSTM(2*d2, d2)



	def forward(self, passes=1):
		"""
		Forward function of the FusionBlock.

		Performs a number of passes through the fusion block.
		Uses the tok2ent, graph_attention, bidaf (from utils) and 
		graph2doc functions to do this.
		At each pass, we get updated context and query embeddings.
		The context and query embeddings obtained in the last pass
		are returned.

		:param passes: number of passes through the FusionBlock,
					   default is 1, experiments in the paper use 2
		:return Ct: updated context embedding (M, d2)
		:return query_emb: updated query embedding (L, d2)
		"""
		for p in range(passes):
			self.entity_embs = self.tok2ent() # (N, 2d2)
			self.entity_embs = self.entity_embs.unsqueeze(2) # (N, 2d2, 1)
			updated_entity_embs = self.graph_attention() # (N, d2)

			# the second one is updated; that's why it's the other way round as in the DFGN paper
			self.query_emb = self.bidaf(updated_entity_embs, self.query_emb) # (N, d2) formula 9

			Ct = self.graph2doc(updated_entity_embs) # (M, d2)
			self.context_emb = Ct # update the context embeddings for the next pass
			#print(f"pass {p}:\n{Ct}\n")#CLEANUP

		return Ct, self.query_emb


	def tok2ent(self):
		"""
		Document to Graph Flow from the paper (section 3.4, paragraph 2)

		Obtain the embedding of the entities from the context embeddings.
		Both mean-pooling and max-pooling are applied.

		:return entity_emb: (N, 2d2) entity embeddings obtained from context embeddings
		"""
		M = self.context_emb.shape[0]
		N = self.bin_M.shape[1]
		#print(f"context_emb: {self.context_emb.shape}")#CLEANUP
		#print(f"bin_M: {self.bin_M.shape}")  # CLEANUP
		entity_emb = self.context_emb.unsqueeze(1).expand(-1, N, -1) # (M, N, d2)
		#print(f"entity_emb1: {entity_emb.shape}")  # CLEANUP

		bin_M_prime = self.bin_M.unsqueeze(2) # (M, N, 1)
		#print(f"bin_M_prime: {bin_M_prime.shape}")  # CLEANUP

		entity_emb = entity_emb * bin_M_prime # (M, N, d2) * (M, N 1) = (M, N, d2)
		#print(f"entity_emb2: {entity_emb.shape}")  # CLEANUP

		entity_emb = entity_emb.permute(1, 2, 0) # (M, N, d2) -> (N, d2, M)
		#print(f"entity_emb3: {entity_emb.shape}")  # CLEANUP

		# For the next lines: (N, d2, M) -> (N, d2, 1) -> (N, d2)
		mean_pooling = F.avg_pool1d(entity_emb, kernel_size=M).squeeze(-1)
		max_pooling = F.max_pool1d(entity_emb, kernel_size=M).squeeze(-1)

		entity_emb = torch.cat((mean_pooling, max_pooling), dim=-1) # (N, 2d2)
		#print(f"entity_emb4: {entity_emb.shape}")  # CLEANUP

		return entity_emb # (N, 2d2)

	def graph_attention(self):
		"""
		This implements Dynamic Graph Attention (section 3.4, paragraph 3).
		Each node of the entity graph propagates information
		to its neighbors in order to produce updated entity
		embeddings.

		:return: E_t: (N, d2) updated entity embeddings 
		"""
		#TODO avoid for-loops, but first make the method run.
		#TODO avoid torch.Tensor where possible.
		#TODO change all this to comply with batches! But before, think about the structure of this whole module.
		N = self.entity_embs.shape[0] # number of entities, taken from  (N, 2d2, 1)
		assert N == len(self.graph.graph) # CLEANUP? # N should be equal to the number of graph nodes
		
		# formula 1 # (L, d2) --> (1, L, d2) --> (1, d2, L) --> (1, d2, 1)
		q_emb = F.avg_pool1d(self.query_emb.unsqueeze(0).permute(0, 2, 1),
							 kernel_size=self.query_emb.shape[0])
		q_emb = q_emb.permute(0, 2, 1).squeeze(0) # (1, 1, d2) --> (1, d2)

		# N * ( (1, d2) x (d2, 2d2) x (2d2, 1) ) --> (N, 1, 1) # formula 2
		gammas = torch.tensor([ torch.chain_matmul(q_emb, self.V, e)/self.droot for e in self.entity_embs ]) #TODO avoid for-loop and torch.tensor()
		mask = torch.sigmoid(gammas)   # (N, 1, 1) # formula 3
		E = torch.stack([m*e for m,e in zip(mask, self.entity_embs.T)])  # (N, 1, 2d2) # formula 4
		E = E.squeeze(1).T # (N, 2d2) --> (2d2, N) #TODO do we really need to squeeze?


		""" disseminate information across the dynamic sub-graph """
		betas = torch.zeros(N, N)
		alphas = torch.zeros(N, N) # scores of how much information flows from i to the j

		# N times [(d2, 2d2) * (2d2, 1)] --> (N, d2, 1)  # formula 5
		hidden = torch.stack([torch.matmul(self.U,e) + self.b for e in E]) # TODO avoid the for-loop

		for i, h_i in enumerate(hidden): # h_i.shape = (d2, 1) #TODO try to avoid these for-loops
			for j, rel_type in self.graph.graph[i]["links"]: # only for neighbor nodes
				pair = torch.cat((h_i, hidden[j])) # (2d2, 1)
				betas[i][j] = F.leaky_relu(torch.matmul(self.W.T, pair)) # formula 6

				# avoid overflow errors for too big values
				exes = [] #TODO how to handle overflows? With nan values?
				for j in range(N):
					try:
						exes.append(exp(betas[i][j]))
					except OverflowError:
						exes.append(float('inf'))
				sumex = sum(exes)

			for j in range(N): # compute scores for all node combinations
				try:
					alphas[i][j] =  exp(betas[i][j]) / sumex # formula 7
				except OverflowError:
					alphas[i][j] = float('inf')/sumex

		""" compute total information received per node """
		E_t = [] #really N * (d2, 1)?

		for i in range(N):
			# scalar * (j, d2, 1) --> sum --> (d2, 1)
			score_sum = sum([alphas[j][i] * hidden[j] for j, rel_type in self.graph.graph[i]["links"]])
			# --> relu --> (d2, 1)
			E_t.append(F.relu(score_sum)) # formula 8

		return torch.stack(E_t).squeeze(dim=-1) # (N, d2) #TODO avoid torch.Tensor()

	def graph2doc(self, entity_embs):
		"""
		#TODO update docstring
		This implements Graph to Document Flow (section 3.4, last paragraph).

		Given the updated entity embeddings, using the same binary matrix
		as in tok2ent, produce the updated context embeddings.

		:param entity_embs: (N, d2) updated entity embeddings as obteined
									from graph_attention
		:return output: (M, d2) updated context embeddings
		"""

		# unsqueeze to represent the batch
		emb_info = torch.matmul(self.bin_M, entity_embs).unsqueeze(0) # (M, N) x (N, d2) -> (M, d2) -> (1, M, d2)
		input = torch.cat((self.context_emb.unsqueeze(0), emb_info), dim=-1) # (1, M, 2d2)
		output, hidden_states = self.g2d_layer(input) # (1, M, d2) # formula 10

		return output.squeeze(0)










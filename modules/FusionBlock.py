"""
This implements the Fusion block from the paper (Section 3.4)
"""
import numpy #TODO?
import torch

class FusionBlock():
	"""
	"""

	def __init__(self, context_emb, query_emb, bin_M):
		self.context_emb = context_emb
		self.query_emb = query_emb
		self.bin_M = bin_M


	def tok2ent():
		"""
		Document to Graph Flow from the paper

		:param context_emb: M x d_2 (context embedding from Encoder)
		:param bin_M: M x N (binary matrix from EntityGraph)
		:return : N x 2d_2
		"""
		M = self.context_emb.shape[0]
		N = bin_M.shape[1]

		entity_emb = self.context_emb.expand(-1, N, -1) # M x N x d_2
		bin_M_prime = self.bin_M.unsqueeze(2) # M x N x 1
		entity_emb = entity_emb * bin_M_prime # M x N x d_2
		entity_emb = entity_emb.permute(1, 2, 0) # M x N x d_2 -> N x d_2 x M

		# For the next two lines: N x d_2 x M -> N x d_2 x 1 -> N x d_2
 		mean_pooling = torch.nn.functional.avg_pool1d(entity_emb, kernel_size=M).squeeze(-1) 
		max_pooling = torch.nn.functional.max_pool1d(entity_emb, kernel_size=M).squeeze(-1)

		entity_emb = torch.cat((mean_pooling, max_pooling), dim=-1) # N x 2d_2

		return entity_emb
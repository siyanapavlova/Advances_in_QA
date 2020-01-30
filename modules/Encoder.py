"""
This module implements the Encoder from the paper (Section 3.3)
"""

import torch
from transformers import BertTokenizer, BertModel

class Encoder():
	"""
	TODO: write docstring
	"""

	def __init__(self, query=None, context=None):
		"""
		TODO: write docstring
		"""
		if query:
			self.query = query
		else:
			print("No query for Encoder. Working with toy example.")
			self.query = "Who had a little lamb?"

		if context:
			self.context = context
		else:
			print("No context for Encoder. Working with toy example.")
			self.context = [
				["Mary and her lamb",
					["Mary had a little lamb.",
					 "The lamb was called Tony.",
					 "Some Microsoft executives wanted to hire Tony."]],
				["All like it but John",
					["Siyana thought that Tony is cute.",
					 "Well, I also think that he is nice.",
					 "Mary, however liked Tony even more than we do."]]
			]

		#TODO does this really flatten into a single string?
		self.concatenated = query + ' '.join([' '.join(para[1]) for para in context])

	def encode(self,
			   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'),
			   model = BertModel.from_pretrained('bert-base-uncased',
												output_hidden_states=True,
												output_attentions=True)):
		"""
		TODO: write docstring
		"""
		input_ids = torch.tensor([tokenizer.encode(self.concatenated)])
		all_hidden_states, all_attentions = model(input_ids)[-2:]

		# This is the embedding of the context + query
		# [-1] stands for the last hidden state
		# [0] is the first (and only) sentence
		# sentence being defined as a sequence of characters, and not a linguistic sentence)
		return all_hidden_states[-1][0]

	class BiDAFNet(torch.nn.Module):
		"""
		TODO: write docstring

		BiDAF paper: arxiv.org/pdf/1611.01603.pdf
		There's a link to the code, but that uses TensorFlow
		"""
		def __init__(self, input_size=768, output_size=300):
			pass


# Advances_in_QA: Re-implementing 'Dynamically Fused Graph Networks'

Here are exercises and the final project from the "Advances in QA" class, winter semester 2019/2020, Uni Saarland. 

The final project aims to re-implement the system published by [Xiao et al. (2019)](https://arxiv.org/pdf/1905.06933.pdf "pdf on arxiv.org"), *"Dynamically Fused Graph Network for Multi-hop Reasoning"* with as little assistance from their [openly available code](https://github.com/woshiyyya/DFGN-pytorch "Github repository") as possible.

#### DFGN in Short
Multi-hop question answering (QA) requires a system to derive the answer to a question from multiple text resources which, each on its own, don't contain the full answer.

In short, the workflow of dynamically fused graph networks (DFGN) is as follows: select several relevant paragraphs, construct an entity graph from them, and then look at parts of this graph to compute which entities contribute the most to answering the question at hand. Perform this step multiple times (each time looking at different parts of the entity graph) and take into account the entities' contribution from the previous iteration. This way, the graph network converges to the final answer.

The intuition is that with multiple iterations, relevant entities propagate their importance to other, directly connected entities. The entity graph has no document boundaries, enabling free flow of information ( = reasoning) across paragraphs. By focusing on sub-parts of the entity graph, **[TODO] why only look at subgraphs?** 


#### External and Internal Modules
- `utils.py` (local) provides helper functions and classes.
- `tqdm` for progress bars. [(get it here)](https://tqdm.github.io/ "Github")
- `flair` for named entity recognition (NER) [(get it here)](https://github.com/flairNLP/flair)
- `pycorenlp.StanfordCoreNLP` for NER (not necessary)s [(get it here)](https://stanfordnlp.github.io/CoreNLP/other-languages.html)
- `transformers` by Huggingface, supplying BERT [(get it here)](https://github.com/huggingface/transformers#installation)



#### DFGN in Detail

The paper describes an architecture which is split into 5 modules:
1) **paragraph selector** returns only the most relevant paragraphs (= "the context")
2) **graph constructor** returns an entity graph from the context
3) **encoder** uses BERT to encode the context and the question
4) **fusion block** – the heard of the DFGN – looks at parts of the entity graph for several iterations
5) **LSTM prediction layer** takes the fusion block's output and returns the final answer


### To run the paragraph selector
python train_ps.py config/train_ps_80-20.cfg paragraph-selector

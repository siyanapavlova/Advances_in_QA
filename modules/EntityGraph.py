"""
This class implements the Entity Graph Constructor from the paper, section 3.2
"""

import sys
from pycorenlp import StanfordCoreNLP #CLEANUP when not calling the server anymore
from transformers import BertTokenizer
from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np

class EntityGraph():
    """
    Make an entity graph from a context (i.e., a list of paragraphs (i.e., a list
    of sentences)). This uses either flair (default) or StanfordCoreNLP for NER
    and subsequently connects them via 3 types of relations.

    The graph is implemented as a dictionary of node IDs to nodes.
    Node IDs are given in a counting manner: earlier entities have smaller IDs.
    A node in the graph is a dictionary:
    'address':      tuple(int, int, int, int) -- (paragraph, sentence, start, end)
    'context_span': tuple(int, int) -- (absolute_start, absolute_end)
    'token_ids':    list[int] -- [token_number(s)]
    'links':        list[tuple(int, int)] -- [(related_node_ID, relation_type)]
    'mention':      str --'Enty McEntityface'

    Relation types are encoded as integers 0, 1, and 2:
    0 - sentence-level links
    1 - context-level links
    2 - paragraph-level links

    Additionals:
    The graph object is initialized with a BertTokenizer object.
    The object stores the context in structured form ans as token list.
    The binary matrix for tok2ent is created upon initialization.
    A call to the object with one or more IDs will return a sub-graph.
    """

    def __init__(self, context=None, tagger='flair', max_nodes=40):
        """
        Initialize a graph object with a 'context'.
        A context is a list of paragraphs and each paragraph is a 2-element list
        where the first element is the paragraph's title and the second element
        is a list of the paragraph's sentences.
        Graph nodes are identified by NER; either by flair or by StanfordCoreNLP.

        :param context: one or more paragraphs of text
        :type context: list[ list[ str, list[str] ] ]
        :param tagger: one of 'flair' or 'stanford'
        :type tagger: str
        :type max_nodes: int
        """
        if context:
            self.context = context
        else:
            print("No context for GraphConstructor. Working with toy example.")
            self.context = [
                ["Mary and her lamb",
                 ["Mary had a little lamb.",
                  " The lamb was called Tony.",
                  " One day, Bill Gates wanted to hire Tony."]],
                ["All like it but John",
                 ["Siyana thought that Tony is cute.",
                  " Well, I also think that he is nice.",
                  " Mary, however liked Tony even more than we do."]]
            ]

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokens = self.tokenizer.tokenize(self.flatten_context())

        self.graph = {}
        self.discarded_nodes = {}

        self._find_nodes(tag_with=tagger)
        self._connect_nodes()
        self._add_entity_spans()
        self.prune(max_nodes) # requires entity links
        self.M = self.tok2ent(add_token_mapping_to_graph=True)

    def __repr__(self):
        result = f""
        for id, node in self.graph.items():
            result += f"{id}\n" + \
                      f"   mention:      {node['mention']}\n"+ \
                      f"   address:      {node['address']}\n"+ \
                      f"   context_span: {node['context_span']}\n" + \
                      f"   token_ids:    {node['token_ids']}\n" + \
                      f"   links:        {node['links']}\n"
        return result.rstrip()

    def __call__(self, *IDs):
        """
        Return the subgraph of nodes corresponding to the given IDs,
        or {"INVALID_ID":id} if an ID is invalid.
        :param IDs: IDs of nodes of the graph (int)
        :return: dictionary of IDs to graph nodes
        """
        result = {}
        for i in IDs:
            if i in self.graph:
                result[i] = self.graph[i]
            else:
                result[i] = {"INVALID_ID":i}
        return result

    def _find_nodes(self, tag_with='flair'):
        """
        Apply NER to extract entities and their positional information from the
        context.
        When working with flair, a heuristic is used to counteract cases
        in which an entity contains trailing punctuation (this would conflict
        with BertTokenizer later on).
        :param tag_with: one of 'stanford' or 'flair'
        :type tag_with: str
        """
        ent_id = 0

        if tag_with == 'stanford':
            tagger = StanfordCoreNLP("http://corenlp.run/")
            for para_id, paragraph in enumerate(self.context):  # between 0 and 10 paragraphs
                sentences = [paragraph[0]] + paragraph[1] # merge header and sentences to one list
                for sent_id, sentence in enumerate(sentences):  # first sentence is the paragraph title
                    print("NER of", sent_id, "-", sentence) #CLEANUP
                    annotated = tagger.annotate(sentence,
                                             properties={"annotators": "ner",
                                                         "outputFormat": "json"})
                    entities = annotated['sentences'][0]['entitymentions'] # list of dicts

                    for e in entities:
                        self.graph[ent_id] = {"address":(para_id,
                                                         sent_id,
                                                         e['characterOffsetBegin'],
                                                         e['characterOffsetEnd']),
                                              "links":[], # relations
                                              "mention":e['text'] # name of the node
                                             }
                        ent_id += 1

        elif tag_with == 'flair':
            tagger = SequenceTagger.load('ner')
            for para_id, paragraph in enumerate(self.context):  # between 0 and 10 paragraphs
                # merge header and sentences to one list and convert to Sentence object
                sentences = [Sentence(s) for s in [paragraph[0]] + paragraph[1]]
                tagged_sentences = tagger.predict(sentences)

                for sent_id, sentence in enumerate(tagged_sentences):  # first sentence is the paragraph title
                    entities = sentence.get_spans('ner')
                    for e in entities:
                        if e.text.endswith(('.','?','!',',',':')): # counter tagging errors
                            end_pos = e.end_pos - 1
                            text = e.text[:-1]
                        else:
                            end_pos = e.end_pos
                            text = e.text
                        self.graph.update(
                            {ent_id : {"address": (para_id,
                                                   sent_id,
                                                   e.start_pos,
                                                   end_pos),
                                       "links": [],  # relations
                                       "mention": text  # name of the node
                                       }
                            })
                        ent_id += 1

        else:
            print(f"ERROR: invalid tagger {tag_with}. Continuing with 'flair'")
            self._find_nodes(tag_with='flair')

    def _connect_nodes(self):
        """
        Establish sentence-level, context-level, and paragraph-level links.
        All 3 relation types are symmetric, but stored in both of any two
        related nodes under 'links'. A node containing the tuple (i,r) has a
        relation of type r to the node with ID i.
        Relation types are marked by integer values 0, 1, and 2:
        0 = Sentence-level links
        1 = context-level links
        2 = paragraph-level links
        """
        # all relations are symmetric -> they're always added to both nodes
        title_entities = {}
        paragraph_entities = {}
        for k,e in self.graph.items():
            if e['address'][1] == 0:
                title_entities[k] = e
            else:
                paragraph_entities[k] = e

        for k1,e1 in paragraph_entities.items(): # look at all nodes in paragraphs
            for k2,e2 in paragraph_entities.items():
                if k2 > k1: # only match up with subsequent nodes
                    # same paragraph and sentence IDs -> sentence-level link
                    if e1['address'][0] == e2['address'][0] and \
                       e1['address'][1] == e2['address'][1]:
                        self.graph[k1]["links"].append((k2, 0))
                        self.graph[k2]["links"].append((k1, 0))
                    # same name -> context-level link
                    if e1['mention'] == e2['mention']:
                        self.graph[k1]["links"].append((k2, 1))
                        self.graph[k2]["links"].append((k1, 1))

        for k1,e1 in title_entities.items(): # paragraph-level links
            for k2,e2 in paragraph_entities.items():
                if e1['address'][0] == e2['address'][0]: # same paragraph
                    self.graph[k1]["links"].append((k2, 2))
                    self.graph[k2]["links"].append((k1, 2))

    def _add_entity_spans(self):
        """
        Map each entity onto their character span at the scope of the whole
        context. This assumes that each sentence/paragraph is separated with
        one whitespace character.
        :return: dict{entityID:(start_pos,end_pos)}
        """
        abs_spans = {} # {entity_ID:(abs_start,abs_end)}
        list_context = [[p[0]] + p[1] for p in self.context]  # squeeze header into the paragraph
        node_IDs = sorted(self.graph.keys())  # make sure that the IDs are sorted
        cum_pos = 0  # cumulative position counter (gets increased with each new sentence)
        prev_sentnum = 0

        for id in node_IDs:  # iterate from beginning to end
            para, sent, rel_start, rel_end = self.graph[id]['address']
            if sent != prev_sentnum:  # we have a new sentence!
                # increase accumulated position by sent length plus a space
                cum_pos += len(list_context[para][prev_sentnum]) + 1

            abs_start = rel_start + cum_pos
            abs_end = rel_end + cum_pos
            abs_spans[id] = (abs_start, abs_end)

            prev_sentnum = sent

        # add the information to the graph nodes
        for id, (start, end) in abs_spans.items():
            self.graph[id].update({"context_span": (start, end)})

    def tok2ent(self, add_token_mapping_to_graph=False):
        #TODO rename this to avoid confusion with FusionBlock.tok2ent()!
        """
        Create a mapping (and subsequently, the matrix M) from entity IDs to
        token IDs, having used BertTokenizer for tokenization. If specified,
        the mapping is added to the graph's nodes (under the key 'token_ids').
        :return: numpy ndarray of shape (#tokens, #entities) -- the matrix M
        """

        """ preparations """
        # set up the variables for the loop
        entity_stack = sorted([(id, node['mention']) for id,node in self.graph.items()])
        in_ent = False
        accumulated_string = ""
        acc_count = 0
        # prepare the first entity
        entity = entity_stack.pop(0) # tuple: (ID, entity_string)
        entity = (entity[0], entity[1].lower().split())
        assert type(entity[1]) is list

        mapping = {}  # this will contain the result:  {ID:[token_nums]}

        """ map node IDs to token indices """
        for i, t in enumerate(self.tokens):

            if t.startswith("##"): # append the wordpiece to the previous token
                accumulated_string += t.strip("#") # add the current token, but without '##'
                acc_count += 1
            else: # nothing special happens.
                accumulated_string = t
                acc_count = 1

            if in_ent and t not in entity[1]: # switch back to out-of-entity mode
                in_ent = False
                if entity_stack:
                    entity = entity_stack.pop(0) # fetch the next entity
                    entity = (entity[0], entity[1].lower().split())
                    assert type(entity[1]) is list

            if accumulated_string in entity[1]:
                # add all the accumulated token positions to the entity's entry
                if entity[0] not in mapping: # new entry with the ID as key
                    mapping[entity[0]] = [i-acc for acc in range(acc_count)]
                else:
                    mapping[entity[0]].extend([i-acc for acc in range(acc_count)])
                in_ent = True # we may be inside a multi-word entity

        mapping = {k:sorted(v) for k,v in mapping.items()} # sort values

        # add the mapping of entity to token numbers to the graph's nodes
        if add_token_mapping_to_graph:
            for id in self.graph:
                self.graph[id].update({"token_ids":mapping[id]})

        """ create binary matrix from the mapping """
        M = np.zeros((len(self.tokens), len(mapping)))
        for node,tokens in mapping.items():
            for tok in tokens:
                M[tok][node] = 1

        return M

    def flatten_context(self, siyana_wants_a_oneliner=False):
        """
        return the context as a single string,
        :return: string containing the whole context
        """

        if siyana_wants_a_oneliner:  # This is for you, Siyana!
            return " ".join([p[0] + " " + " ".join(["".join(s) for s in p[1:]]) for p in self.context])

        final = ""
        for para in self.context:
            for sent in para:
                if type(sent) == list:
                    final += "".join(sent) + " "
                else:
                    final += sent + " "
        final = final.rstrip()
        return final

    def prune(self, max_nodes):
        """
        Limit the number of nodes in a graph by deleting the least connected
        nodes ( = smallest number of link tuples). If two nodes have the same
        number of connections, the one with the higher ID gets deleted.
        Pruned nodes are stored in a separate data structure (just in case)
        :param max_nodes: maximum number of nodes
        """
        if len(self.graph) > max_nodes:
            # temporary representation, sorted by number of connections
            deletable_keys = sorted(self.graph,
                                    key=lambda x: len(self.graph[x]['links']),
                                    reverse=True
                                    )[max_nodes:] # from max_nodes to the end
            for node in deletable_keys:
                self.discarded_nodes[node] = self.graph[node] # add the discarded node
                del self.graph[node]

    def relation_triplets(self):
        """
        Computes the set of relation triplets (e1, e2, rel_type) of a graph,
        where e1 and e2 are two related entities and rel_type is their relation.
        All 3 relation types are symmetric and are represented as two
        one-directional edges in the EntityGraph object, but here only one of
        a relation's two edges is included.
        Relation types are coded as:
        0 - sentence-level link
        1 - context-level link
        2 - paragraph-level link
        :return: set of link triplets (e1, e2, rel_type)
        """
        relations = set()
        for id,node in self.graph.items(): # get all relations (both directions)
            relations.update(set([ (id,r[0],r[1]) for r in node['links'] ]))

        result = set()
        for e1,e2,rt in relations:
            if (e2,e1,rt) not in result: # only keep one of the two triplets
                result.add((e1,e2,rt))
            else:
                pass
        return result

    def avg_degree(self):
        """
        number of average connections per node (bidirectional links count only once)
        :return: average degree of the whole graph
        """
        return len(self.relation_triplets())/len(self.graph)

    def visualize(self, This_method_doesnt_work): #CLEANUP?
        #TODO implement visualization code?
        # https://www.data-to-viz.com/graph/network.html
        """
        node labels: entity name (paragraph, sentence)
        edges color-coded


        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        import pandas as pd
        import networkx as nx
        from modules.EntityGraph import EntityGraph

        color_codes = {0: "blue", 1: "green", 2: "red"}
        g = EntityGraph()
        connections = [str(c[0]) + " " +
                       str(c[1]) + " " +
                       "{'color':'" + color_codes[c[2]] + "'}" for c in
                            g.relation_triplets()]
        G = nx.parse_edgelist(connections, nodetype=int)
        nx.draw(G)
        plt.show()
        """
        pass
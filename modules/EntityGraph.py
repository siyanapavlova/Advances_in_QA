"""
This class implements the Entity Graph Constructor from the paper, section 3.2
"""

from pycorenlp import StanfordCoreNLP #CLEANUP when not calling the server anymore

class EntityGraph():
    """
    Make an entity graph from a context (i.e., a list of paragraphs (i.e., a list
    of sentences)). This uses StanfordCoreNLP to extract named entities and
    subsequently connects them via 3 types of relations.

    The paragraph is implemented as a dictionary of node IDs to nodes.
    Node IDs are given in a counting manner: earlier entities have smaller IDs.
    A node in the graph is a 6-tuple:
    0 - int - node ID
    0 - int - paragraph number
    1 - int - sentence number (first sentence = paragraph title)
    2 - int - start index
    3 - int - end index
    4 - list - relations (= tuples of (ID, relation_type))
    5 - str - entity (substring of the context defined by 1 through 4)

    Relation types are encoded as integers 0, 1, and 2:
    0 - sentence-level links
    1 - context-level links
    2 - paragraph-level links
    """

    def __init__(self, context=None, max_nodes=40):
        """
        Initialize a graph object with a 'context'.
        A context is a list of paragraphs.
        Each paragraph is a 2-element lists where the first element is the
        paragraph's title and the second element is a list of the paragraph's
        sentences.
        :param context: one or more paragraphs of text
        :type context: list[ list[ str, list[str] ] ]
        """
        if context:
            self.context = context
        else:
            print("No context for GraphConstructor. Working with toy example.")
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
        self.graph = {}
        self.discarded_nodes = {}
        self.find_nodes()
        self.connect_nodes()
        self.prune(max_nodes)

    def __repr__(self):
        return "\n".join([str(k)+"   "+str(v) for k,v in self.graph.items()])

    def __call__(self, *IDs):
        """
        Return the subgraph of nodes corresponding to the given IDs,
        or "INVALID_ID" if an ID is invalid.
        :param IDs: IDs of nodes of the graph (int)
        :return: dictionary of IDs to graph nodes
        """
        result = {}
        for i in IDs:
            if i in self.graph:
                result[i] = self.graph[i]
            else:
                result[i] = "INVALID_ID"
        return result

    def find_nodes(self):
        """
        Extracts named entities (using StanfordCoreNLP for NER) to the graph
        data structure.
        """
        # TODO change from calling a server to calling a local system
        nlp = StanfordCoreNLP("http://corenlp.run/")
        ent_id = 0
        for para_id, paragraph in enumerate(self.context):  # between 0 and 10 paragraphs
            sentences = [paragraph[0]]
            sentences.extend(paragraph[1])
            #print("\n".join([str(i)+"   "+str(s) for i,s in enumerate(sentences)])) #CLEANUP
            for sent_id, sentence in enumerate(sentences):  # first sentence is the paragraph title
                print("NER of", sent_id, "-", sentence) #CLEANUP
                annotated = nlp.annotate(sentence,
                                     properties={"annotators": "ner",
                                                 "outputFormat": "json"})
                try:
                    entities = annotated['sentences'][0]['entitymentions'] # list of dicts
                except TypeError as e: #CLEANUP
                    print(e)
                    print(annotated)

                for e in entities:
                    ent_id += 1
                    self.graph[ent_id] = (para_id,
                                          sent_id,
                                          e['characterOffsetBegin'],
                                          e['characterOffsetEnd'],
                                          [], # relations
                                          e['text'] # name of the node
                                         )

    def connect_nodes(self):
        """
        Establish sentence-level, context-level, and paragraph-level links.
        All 3 relation types are symmetric, but stored in both of any two
        related nodes. A node containing the tuple (i,k) has a relation of
        type k to the node with ID i.
        Relation types are marked by integer values 0, 1, and 2:
        0 = Sentence-level links
        1 = context-level links
        2 = paragraph-level links
        """

        # all relations are symmetric -> they're always added to both nodes
        title_entities = {}
        paragraph_entities = {}
        for k,e in self.graph.items():
            if e[1] == 0:
                title_entities[k] = e
            else:
                paragraph_entities[k] = e

        for k1,e1 in paragraph_entities.items(): # look at all nodes in paragraphs
            for k2,e2 in paragraph_entities.items():
                if k2 > k1: # only match up with subsequent nodes
                    # same paragraph and sentence IDs -> sentence-level link
                    if e1[0] == e2[0] and e1[1] == e2[1]:
                        self.graph[k1][4].append((k2, 0))
                        self.graph[k2][4].append((k1, 0))
                    # same name -> context-level link
                    if e1[5] == e2[5]:
                        self.graph[k1][4].append((k2, 1))
                        self.graph[k2][4].append((k1, 1))

        for k1,e1 in title_entities.items(): # paragraph-level links
            for k2,e2 in paragraph_entities.items():
                if e1[0] == e2[0]: # same paragraph
                    self.graph[k1][4].append((k2, 2))
                    self.graph[k2][4].append((k1, 2))

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
                                    key=lambda x: len(self.graph[x][4]),
                                    reverse=True
                                    )[max_nodes:] # from max_nodes to the end
            for node in deletable_keys:
                self.discarded_nodes[node] = self.graph[node]
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
            relations.update(set([ (id,r[0],r[1]) for r in node[4] ]))

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
        return len(self.link_triplets())/len(self.graph)

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
                            g.link_triplets()]
        G = nx.parse_edgelist(connections, nodetype=int)
        nx.draw(G)
        plt.show()
        """
        pass
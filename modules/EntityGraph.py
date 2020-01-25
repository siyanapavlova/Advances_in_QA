"""
This class implements the Entity Graph Constructor from the paper, section 3.2
"""

from pycorenlp import StanfordCoreNLP #CLEANUP when not calling the server anymore

class EntityGraph():
    """
    Make an entity graph from a context (i.e., a list of paragraphs (i.e., a list
    of sentences)). This uses StanfordCoreNLP to extract named entities and
    subsequently connects them via 3 types of relations.

    A node in the graph is a 7-tuple:
    0 - int - node ID
    1 - int - paragraph number
    2 - int - sentence number (first sentence = paragraph title)
    3 - int - start index
    4 - int - end index
    5 - list - relations (= tuples of (ID, relation_type))
    6 - str - entity (substring of the context defined by 1 through 4)

    Relation types are encoded as integers 0, 1, and 2:
    0 - sentence-level links
    1 - context-level links
    2 - paragraph-level links
    """

    def __init__(self, context=None, max_nodes=40):
        """
        Initialize a graph object with a 'context'.
        :param context: one or more paragraphs of text
        :type context: list[list[str]] list of lists of strings
        :type context:
        #TODO like this: [
                             ["title phrase", ["first sentence", "second sentence"]],
                             ["second par. title", ["...", "..."]]
                         ]
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
        self.graph = [] # list characteristics are utilized by connect_graph()
        self.discarded_nodes = []
        self.find_nodes()
        self.connect_nodes()
        self.prune(max_nodes)

    def __repr__(self):
        return "\n".join([str(t) for t in self.graph])


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
                    # TODO change relations container from list to set?
                    # TODO if so, then also change connect_graph()!
                    self.graph.append((ent_id,
                                       para_id,
                                       sent_id,
                                       e['characterOffsetBegin'],
                                       e['characterOffsetEnd'],
                                       [], # relations
                                       e['text'] # name of the node
                                      ))
                    ent_id += 1

    def connect_nodes(self):
        """
        Establish sentence-level, context-level, and paragraph-level links.
        All 3 relation types are symmetric, but stored in both of any two
        related nodes.
        """

        # all relations are symmetric -> they're always added to both nodes
        title_entities = [e for e in self.graph if e[2]==0]
        paragraph_entities = [e for e in self.graph if e not in title_entities]

        for e1 in paragraph_entities: # look at all nodes in paragraphs
            for e2 in paragraph_entities:
                if e2[0] > e1[0]: # only match up with subsequent nodes
                    # same paragraph and sentence IDs -> sentence-level link
                    if e1[1] == e2[1] and e1[2] == e2[2]:
                        self.graph[e1[0]][5].append((e2[0], 0))
                        self.graph[e2[0]][5].append((e1[0], 0))
                    # same name -> context-level link
                    if e1[6] == e2[6]:
                        self.graph[e1[0]][5].append((e2[0], 1))
                        self.graph[e2[0]][5].append((e1[0], 1))

        for e1 in title_entities: # paragraph-level links
            for e2 in paragraph_entities:
                if e1[1] == e2[1]: # same paragraph
                    self.graph[e1[0]][5].append((e2[0], 2))
                    self.graph[e2[0]][5].append((e1[0], 2))

    def prune(self, max_nodes):
        #TODO test this method
        """
        #TODO docstring
        :param max_nodes:
        :return:
        """
        if len(self.graph) > max_nodes:
            # temporary representation, sorted by number of connections
            pruned_graph = sorted(self.graph,
                                  key=lambda x:len(x[5]),
                                  reverse=True)[:max_nodes]
            #TODO why does prune(8) lead to 9 nodes?
            for node in self.graph: # execute pruning
                if node not in pruned_graph:
                    #TODO do we really need discarded_nodes?
                    self.discarded_nodes.append(node) # keep discarded nodes, just in case
                    self.graph.remove(node)
        else:
            pass

    def link_triplets(self):
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
        for node in self.graph: # get all relations (both directions)
            relations.update(set([ (node[0],r[0],r[1]) for r in node[5] ]))
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


    def visualize(self):
        #TODO implement visualization code?
        # https://www.data-to-viz.com/graph/network.html
        """
        node labels: entity name (paragraph, sentence)
        edges color-coded
        """
        pass
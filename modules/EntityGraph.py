"""
This class implements the Entity Graph Constructor from the paper, section 3.2
"""

from pycorenlp import StanfordCoreNLP


class EntityGraph():
    """
    Make an entity graph from a context (i.e., a list of paragraphs (i.e., a list
    of sentences)). This uses StanfordCoreNLP to extract named entities and
    subsequently connects them via 3 types of relations.

    A node in the graph is a 7-tuple:
    0 - int - node ID
    1 - int - paragraph number
    2 - int - sentence number
    3 - int - start index
    4 - int - end index
    5 - list - relations (= tuples of (ID, relation_type))
    6 - str - entity (substring of the context defined by 1 through 4)

    Relation types are encoded as integers 0, 1, and 2:
    0 - sentence-level links
    1 - context-level links
    2 - paragraph-level links
    """

    def __init__(self, context=None):
        """
        Initialize a graph object with a 'context'.
        :param context: one or more paragraphs of text
        :type context: list[list[str]] list of lists of strings
        """
        if context:
            self.context = context
        else:
            print("No context for GraphConstructor. Working with toy example.")
            self.context = [
                ["Mary had a little lamb.",
                 "The lamb was called Tony.",
                 "Some Microsoft executives wanted to hire Tony."],
                ["Siyana thought that Tony is cute.",
                 "Well, I also think that he is nice.",
                 "Mary, however liked Tony even more than we do."]
            ]
        self.graph = [] # list characteristics are utilized by connect_graph()
        self.populate_graph()
        self.connect_graph()

    def __repr__(self):
        return "\n".join([str(t) for t in self.graph])


    def populate_graph(self):
        """
        Extracts named entities (using StanfordCoreNLP for NER) to the graph
        data structure.
        """
        # TODO change from calling a server to calling a local system
        nlp = StanfordCoreNLP("http://corenlp.run/")
        ent_id = 0
        for para_id, paragraph in enumerate(self.context):  # between 0 and 10 paragraphs
            for sent_id, sentence in enumerate(paragraph):  # usually multiple sentences
                annotated = nlp.annotate(sentence,
                                         properties={"annotators": "ner",
                                                     "outputFormat": "json"})
                entities = annotated['sentences'][0]['entitymentions']
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

    def connect_graph(self):
        """
        Establish sentence-level, context-level, and paragraph-level links.
        All 3 relation types are symmetric, but stored in both of any two
        related nodes.
        """
        """
        As self.graph is a list and IDs are basically counters, the graph nodes
        can be looked at in a "forward-only" fashion so that the algorithm runs
        in sum[i=0..n](i*(i-1)) instead of n^2.
        """
        for e1 in self.graph:
            for e2 in self.graph[e1[0]+1:]: # loop over nodes with higher ID
                # all relations are symmetric -> they're added to both nodes
                if e1[2] == e2[2]:
                    # same sentence ID -> sentence-level link
                    e1[5].append((e2[0], 0))
                    e2[5].append((e1[0], 0))
                if e1[6] == e2[6]:
                    # same name -> context-level link
                    e1[5].append((e2[0], 1))
                    e2[5].append((e1[0], 1))
                if e1[2] == 0 and e1[1] == e2[1]:
                    # e1 in title sent. & same paragraph -> paragraph-level link
                    e1[5].append((e2[0], 2))
                    e2[5].append((e1[0], 2))

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

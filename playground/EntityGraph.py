"""
This class implements the Entity Graph Constructor from the paper, section 3.2
"""

from pycorenlp import StanfordCoreNLP


class EntityGraph():
    """
    #TODO docstring
    Make an entity graph from a context (i.e., a list of paragraphs (i.e., a list
    of sentences). This uses StanfordCoreNLP to extract named entities and
    subsequently connects them via 3 types of relations.

    A node in the graph is a 7-tuple:
    0 - int - node ID
    1 - int - paragraph position
    2 - int - sentence position
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
        #TODO docstring
        :param context:
        """
        if context:
            self.context = context
        else: #CLEANUP
            self.context = [
                ["Mary had a little lamb.",
                 "The lamb was called Tony.",
                 "Some Microsoft executives wanted to hire Tony."],
                ["Siyana thought that Tony is cute.",
                 "Well, I also think that he is nice.",
                 "Mary, however liked Tony even more than we do."]
            ]
        self.graph = []
        self.populate_graph()
        self.connect_graph()


    def __repr__(self):
        return "\n".join([str(t) for t in self.graph])

    def populate_graph(self):
        nlp = StanfordCoreNLP("http://corenlp.run/")
        ent_id = 0
        for para_id, paragraph in enumerate(self.context):  # between 0 and 10 paragraphs
            for sent_id, sentence in enumerate(paragraph):  # usually multiple sentences
                #print(sentence)  # CLEANUP
                annotated = nlp.annotate(sentence,
                                         properties={"annotators": "ner",
                                                     "outputFormat": "json"})
                entities = annotated['sentences'][0]['entitymentions']
                for e in entities:
                    self.graph.append((ent_id,
                                       para_id,
                                       sent_id,
                                       e['characterOffsetBegin'],
                                       e['characterOffsetEnd'],
                                       [],
                                       e['text']
                                       )
                                      )
                    ent_id += 1

    def connect_graph(self):
        #TODO sentence-level links

        #TODO context-level links

        #TODO paragraph-level links

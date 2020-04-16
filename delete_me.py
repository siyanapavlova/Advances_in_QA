context = [
            ["Mary and her lamb",
             ["Mary had a little lamb.",
              " The lamb was called Tony.",
              " One day, Bill Gates wanted to hire Tony."]],
            ["All like it but John",
             ["Siyana thought that Tony is cute.",
              " Well, I also think that he is nice.",
              " Mary, however liked Tony even more than we do."]]
           ]
"""
shape: (question_id, supporting_facts, query, paragraphs, answer),
        where question_id is a string corresponding
              to the datapoint id in HotPotQA
        supporting_facts is a list of strings,
        query is a string,
        paragraphs is a 10-element list where
            the first element is a string
            the second element is a list of sentences (i.e., a list of strings)
"""


 raw_point = ('123456',
 			 ["Mary and her lamb", "All like it but John"],
 			 "Who had a lamb?",

 			 [["Mary and her lamb",
              ["Mary had a little lamb.",
              " The lamb was called Tony.",
              " One day, Bill Gates wanted to hire Tony."]],
              ["All like it but John",
              ["Siyana thought that Tony is cute.",
              " Well, I also think that he is nice.",
              " Mary, however liked Tony even more than we do."]],
              ["All like it but John",
              ["Siyana thought that Tony is cute.",
              " Well, I also think that he is nice.",
              " Mary, however liked Tony even more than we do."]],
              ["Mary and her lamb",
              ["Mary had a little lamb.",
              " The lamb was called Tony.",
              " One day, Bill Gates wanted to hire Tony."]]]
              "Mary"
 			)
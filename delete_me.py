from utils import make_labeled_data_for_predictor
from modules.EntityGraph import EntityGraph

graph = EntityGraph()

raw_point = {"_id":"123456",
 			 "supporting_facts":["All like it but John"],
 			 "question":"Who had a lamb?",

 			 "context": [["Mary and her lamb",
              ["Mary had a little lamb.",
              " The lamb was called Tony.",
              " One day, Bill Gates wanted to hire Tony."]],
              ["This is no a supporting fact",
              ["We just added this paragraph for testing.",
              " We want to see if our label making function works.",
              " We really hope it does."]],
              ["All like it but John",
              ["Siyana thought that Tony is cute.",
              " Well, I also think that he is nice.",
              " Mary, however liked Tony even more than we do."]],
              ["Yet another non-supporting fact",
              ["This one should not have labels of 1 either.",
              " We are nearly there and it's time to test.",
              " Fingers crossed!"]]],
              "answer":"Mary"
 			}

sup_labels, start_labels, end_labels, type_labels = make_labeled_data_for_predictor(graph, raw_point)

print(f"sup_labels: {sup_labels}")
print(f"start_labels: {start_labels}")
print(f"end_labels: {end_labels}")
print(f"type_labels: {type_labels}")

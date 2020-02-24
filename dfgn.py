"""
This script puts together all modules.
"""

import utils
from modules.ParagraphSelector import ParagraphSelector
from modules.EntityGraph import EntityGraph
from modules.Encoder import Encoder


main_cfg_file = utils.loop_input(rtype="filepath", default="config/dfgn.cfg",
                                 msg="Enter configuration file")

cfg = utils.ConfigReader(main_cfg_file)
#cfg.get_param_names() #CLEANUP

dh = utils.HotPotDataHandler(cfg("HotPotQA_filepath"))
data = dh.data_for_paragraph_selector()

ps = ParagraphSelector(model_path=cfg("ps_model_file"),
                       tokenizer=cfg("ps_tokenizer"),
                       encoder_model=cfg("ps_encoder_model"))
enc = Encoder() #TODO fill out this command once Encoder is done!


avg_degrees = [] # for analysis purposes

for datapoint in data:

    """ Paragraph Selector """
    context = ps.make_context(datapoint, threshold=cfg("ps_threshold"))

    """ Graph Constructor """
    graph = EntityGraph(context,
                        tagger=cfg("eg_tagger"),
                        max_nodes=cfg("eg_max_nodes"))
    avg_degrees += graph.avg_degree() # for evaluation purposes

    """ Encoder """
    encoded_context = enc.encode() #TODO call the encoder correctly!

    """ Fusion Block """
    M = graph.M



    """ LSTM Prediction Layer"""







#evaluare the average degree
print(f"overall average degree in the graph: {sum(avg_degrees)/len(avg_degrees)}")
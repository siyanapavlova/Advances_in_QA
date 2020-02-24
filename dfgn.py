"""
This script puts together all modules.
"""

import utils
from modules.ParagraphSelector import ParagraphSelector
from modules.EntityGraph import EntityGraph



main_cfg_file = utils.loop_input(rtype="filepath", default="config/dfgn.cfg",
                                 msg="Enter configuration file")

cfg = utils.ConfigReader(main_cfg_file)
#cfg.get_param_names() #CLEANUP


g =

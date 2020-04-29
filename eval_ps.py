"""
This is for evaluating the paragraph selector.
"""




from utils import Timer
from utils import HotPotDataHandler
from utils import ConfigReader

from modules import ParagraphSelector

import argparse
import sys
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle

# =========== PARAMETER INPUT
take_time = Timer()

parser = argparse.ArgumentParser()
parser.add_argument('config_file', metavar='config', type=str,
                    help='configuration file for evaluation')
parser.add_argument('model_name', metavar='model', type=str,
                    help="name of the model's file")
args = parser.parse_args()
cfg = ConfigReader(args.config_file)

model_abs_path = cfg('model_abs_dir') + args.model_name + "/"
results_abs_path = model_abs_path + args.model_name + ".test_scores"

# check all relevant file paths and directories before starting training
try:
    f = open(cfg("dev_data_abs_path"), "r")
    f.close()
except FileNotFoundError as e:
    print(e)
    sys.exit()

for path in [results_abs_path]:
    if not os.path.exists(path):
        print(f"newly creating {path}")
        os.makedirs(path)

take_time("parameter input")






#TODO load eval data
#TODO load model
#TODO make predictions and evaluate in one step using ps.evaluate()
#TODO output test scores
#TODO take times and output timing into the scores file

print(f"Reading data from {cfg('dev_data_abs_path')}...")  # the whole HotPotQA training set
dh = HotPotDataHandler(cfg("dev_data_abs_path")) # the whole HotPotQA dev set
raw_data = dh.data_for_paragraph_selector() # get raw points

data_limit = cfg("testset_size") if cfg("testset_size") else len(raw_data)







take_time("data loading")








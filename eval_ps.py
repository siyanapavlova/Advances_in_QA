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
predictions_abs_path = cfg('predictions_abs_dir') + args.model_name + ".predicitons"

# check all relevant file paths and directories before starting training
try:
    f = open(cfg("dev_data_abs_path"), "r")
    f.close()
except FileNotFoundError as e:
    print(e)
    sys.exit()

for path in [model_abs_path, cfg('predictions_abs_dir') ]:
    if not os.path.exists(path):
        print(f"newly creating {path}")
        os.makedirs(path)

take_time("parameter input")


print(f"Reading data from {cfg('dev_data_abs_path')}...")  # the whole HotPotQA training set
dh = HotPotDataHandler(cfg("dev_data_abs_path")) # the whole HotPotQA dev set
raw_data = dh.data_for_paragraph_selector() # get raw points

data_limit = cfg("testset_size") if cfg("testset_size") else len(raw_data)

model = ParagraphSelector.ParagraphSelector(model_abs_path) # looks for the 'pytorch_model.bin' in this directory

take_time("data  loading")


print("Evaluating...")
precision, recall, f1, accuracy, ids, y_true, y_pred = model.evaluate(raw_data[:data_limit],
                                                            threshold=cfg("threshold"),
                                                            text_length=cfg("text_length"),
                                                            try_gpu=cfg("try_gpu"))
print("Precision:", precision)
print("Recall:", recall)
print("F score:", f1)
print('----------------------')
take_time("evaluation")

with open(predictions_abs_path, 'w', encoding='utf-8') as f:
    for i in range(len(ids)):
        f.write(ids[i] + "\t" + \
                ','.join([str(int(j)) for j in y_true[i]]) + "\t" + \
                ','.join([str(int(j)) for j in y_pred[i]]) + "\n")

with open(results_abs_path, 'w', encoding='utf-8') as f:
    f.write("Configuration in: " + args.config_file + "\n")
    f.write("Outputs in:  " + predictions_abs_path + \
            "\nPrecision: " + str(precision) + \
            "\nRecall:    " + str(recall) + \
            "\nF score:   " + str(f1) + "\n")
    f.write("Hyper parameters:\n" + str(cfg))

    take_time.total()
    f.write("\n\nTimes taken:\n" + str(take_time))
    print("\ntimes taken:\n", take_time)


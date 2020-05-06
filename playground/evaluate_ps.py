from utils import loop_input
from utils import ConfigReader
from utils import HotPotDataHandler

from modules import ParagraphSelector


import argparse

#TODO CLEANUP this script, because there is evaluate.py, which is more up-to-date
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    
    #=========== PARAMETER INPUT
    config_file = loop_input(rtype=str, default="not specified", #TODO change this to argparse!
                             msg="Configuration file for training")
    model_name = "" #TODO use argparse for this!
    results_name = "" #TODO use argparse for this!

    
    cfg = ConfigReader(config_file)
    verbose = cfg("verbose")
    
    model_abs_path = cfg('model_abs_dir') + model_name
    model_abs_path += '.pt' if not model_name.endswith('.pt') else model_abs_path
    
    
    #========== DATA PREPARATION
    print(f"Reading data from {cfg('data_abs_path')}...")
    dh = HotPotDataHandler(cfg("data_abs_path"))
    data = dh.data_for_paragraph_selector()

    print("Splitting data...")
    train_data_raw, test_data_raw = train_test_split(data,
                                                     test_size=cfg('test_split'),
                                                     random_state=cfg('data_shuffle_seed'),
                                                     shuffle=True)
    
    #========== EVALUATION
    print("Initilising ParagraphSelector...")
    ps = ParagraphSelector(model_path=model_abs_path)
    
    print("Evaluating...")
    precision, recall, f1, ids, y_true, y_pred = ps.evaluate(test_data_raw)
    print('----------------------')
    print("Precision:", precision)
    print("Recall:", recall)
    print("F score:", f1)
    print('----------------------')
    
    
    # TODO: edit the next lines to use the paths from config + whatever
    # is specified by the user (argparse)
    if not os.path.exists(parent_dir + "/models/visuals/"):
        os.makedirs(parent_dir + "/models/visuals/")
    
    with open(parent_dir + '/models/visuals/outputs.txt', 'w', encoding='utf-8') as f:
        for i in range(len(ids)):
            f.write(ids[i] + "\t" + \
                    ','.join([str(int(j)) for j in y_true[i]]) + "\t" + \
                    ','.join([str(int(j)) for j in y_pred[i]]) + "\n")
    
    with open(parent_dir + '/models/visuals/results.txt', 'w', encoding='utf-8') as f:
        f.write("Outputs in: " + parent_dir + '/models/visuals/outputs.txt'+ \
               "\nPrecision: " + str(precision) + \
               "\nRecall: " + str(recall) + \
               "\nF score: " + str(f1))
    
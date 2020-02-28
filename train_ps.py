"""
This script performs training of ParagraphSelector models.
"""

from utils import loop_input
from utils.ConfigReader import ConfigReader
from utils import HotPotDataHandler

from modules import ParagraphSelector

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os,sys,inspect

if __name__ == '__main__':

    #=========== PARAMETER INPUT
    config_file = loop_input(rtype=str, default="not specified", #TODO change this to argparse!
                             msg="Configuration file for training")
    model_name = "" #TODO use argparse for this!
    results_name = "" #TODO use argparse for this!


    cfg = ConfigReader(config_file)
    verbose = cfg("verbose")

    model_abs_path = cfg('model_abs_dir') + model_name
    if model_name.endswith('.pt'):
        model_abs_path += '.pt'

    results_abs_path = cfg('results_abs_dir') + results_name


    #========== DATA PREPARATION
    print(f"Reading data from {cfg('data_abs_path')}...")
    dh = HotPotDataHandler(cfg("data_abs_path"))
    data = dh.data_for_paragraph_selector()

    print("Splitting data...")
    train_data_raw, test_data_raw = train_test_split(data,
                                                     test_size=cfg('test_split'),
                                                     random_state=cfg('data_shuffle_seed'),
                                                     shuffle=True)
    train_data = ParagraphSelector.make_training_data(train_data_raw)
    train_data = shuffle(train_data, random_state=cfg('data_shuffle_seed'))


    #========== TRAINING
    print("Initilising ParagraphSelector...")
    ps = ParagraphSelector()

    print(f"training for {cfg('epochs')} epochs...")
    losses = ps.train(train_data["text"],
                      train_data["label"],
                      cfg("epochs"))

    print(f"Saving model in {model_abs_path}...")
    ps.save(model_abs_path)


    #========== EVALUATION
    print("Evaluating...")
    precision, recall = ps.evaluate(test_data_raw)
    print('----------------------')
    print("Precision:", precision)
    print("Recall:", recall)

    print(f"Saving evaluation results to {}")
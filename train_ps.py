"""
This script performs training of ParagraphSelector models.
"""

from utils import loop_input
from utils.ConfigReader import ConfigReader
from utils import HotPotDataHandler

from modules import ParagraphSelector
from utils import Timer

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os,sys,inspect

if __name__ == '__main__':

    #=========== PARAMETER INPUT
    take_time = Timer()
    config_file = loop_input(rtype=str, default="not specified", #TODO change this to argparse!
                             msg="Configuration file for training")
    model_name = "" #TODO use argparse for this!


    cfg = ConfigReader(config_file)
    verbose = cfg("verbose")

    model_abs_path = cfg('model_abs_dir') + model_name
    model_abs_path += '.pt' if not model_name.endswith('.pt') else model_abs_path
    take_time("parameter input")


    #========== DATA PREPARATION
    print(f"Reading data from {cfg('data_abs_path')}...")
    dh = HotPotDataHandler(cfg("data_abs_path"))
    data = dh.data_for_paragraph_selector()
    take_time("data loading")

    print("Splitting data...")
    train_data_raw, test_data_raw = train_test_split(data,
                                                     test_size=cfg('test_split'),
                                                     random_state=cfg('data_shuffle_seed'),
                                                     shuffle=True)
    train_data = ParagraphSelector.make_training_data(train_data_raw)
    train_data = shuffle(train_data, random_state=cfg('data_shuffle_seed'))
    take_time("data preparation")


    #========== TRAINING
    print("Initilising ParagraphSelector...")
    ps = ParagraphSelector()
    take_time("paragraph selector initiation")

    print(f"training for {cfg('epochs')} epochs...")
    losses = ps.train(train_data["text"],
                      train_data["label"],
                      cfg("epochs"))
    take_time(f"training for {cfg('epochs')} epochs")

    print(f"Saving model in {model_abs_path}...")
    ps.save(model_abs_path)
    take_time("saving model")

    take_time.total()
    print("Times taken:")
    print(take_time)

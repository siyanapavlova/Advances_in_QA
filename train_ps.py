"""
This script performs training of ParagraphSelector models.
"""

from utils import Timer
from utils import HotPotDataHandler
from utils import ConfigReader

from modules import ParagraphSelector

import argparse
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    #=========== PARAMETER INPUT
    take_time = Timer()

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', metavar='config', type=str,
                        help='configuration file for training')
    parser.add_argument('model_name', metavar='model', type=str,
                        help="name of the model's file")
    args = parser.parse_args()

    cfg = ConfigReader(args.config_file)

    model_abs_path = cfg('model_abs_dir') + args.model_name
    model_abs_path += '.pt' if not args.model_name.endswith('.pt') else model_abs_path
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

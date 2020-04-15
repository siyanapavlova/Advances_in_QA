"""
This script performs training of ParagraphSelector models.
"""

from utils import Timer
from utils import HotPotDataHandler
from utils import ConfigReader

from modules import ParagraphSelector

import argparse
import sys
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle


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

    model_abs_path = cfg('model_abs_dir') #+ args.model_name
    #model_abs_path += '.pt' if not args.model_name.endswith('.pt') else ''
    losses_abs_path = cfg("model_abs_dir") + "performance/" + args.model_name + ".losses"
    traintime_abs_path = cfg("model_abs_dir") + "performance/" + args.model_name + ".times"

    take_time("parameter input")


    #========== DATA PREPARATION



    try:
        with open(cfg("pickled_train_data"), "rb") as f:
            train_data = pickle.load(f)
            data_limit = cfg("dataset_size") if cfg("dataset_size") else len(train_data)
        with open(cfg("pickled_dev_data"), "rb") as f:
            dev_data = pickle.load(f)
            dev_data_limit = cfg("dev_data_limit") if cfg("dev_data_limit") else len(dev_data)


    except:
        print(f"Reading data from {cfg('data_abs_path')}...")
        dh = HotPotDataHandler(cfg("data_abs_path"))
        data = dh.data_for_paragraph_selector()
        data_limit = cfg("dataset_size") if cfg("dataset_size") else len(data)

        dev_dh = HotPotDataHandler(cfg("dev_data_abs_path"))
        dev_data = dev_dh.data_for_paragraph_selector()
        dev_data_limit = cfg("dev_data_limit") if cfg("dev_data_limit") else len(dev_data)
        take_time("data loading")

        print("Splitting data...")
        train_data_raw, test_data_raw = train_test_split(data[:data_limit],
                                                         test_size=cfg('test_split'),
                                                         random_state=cfg('shuffle_seed'),
                                                         shuffle=True)
        train_data = ParagraphSelector.make_training_data(train_data_raw,
                                                          text_length=cfg("text_length"))
        #train_data = shuffle(train_data, random_state=cfg('data_shuffle_seed')) #CLEANUP?

        with open(cfg("pickled_train_data"), "wb") as f:
            pickle.dump(train_data, f)

        with open(cfg("pickled_dev_data"), "wb") as f:
            pickle.dump(dev_data, f)



    take_time("data preparation")


    #========== TRAINING
    print("Initilising ParagraphSelector...")
    ps = ParagraphSelector.ParagraphSelector(cfg("bert_model_path"))

    print(f"training for {cfg('epochs')} epochs...")
    losses = ps.train(train_data,
                      dev_data[:dev_data_limit],
                      model_abs_path,
                      epochs=cfg("epochs"),
                      batch_size=cfg("batch_size"),
                      learning_rate=cfg("learning_rate"),
                      eval_interval=cfg("eval_interval"))
    take_time(f"training")

    # print(f"Saving model in {model_abs_path}...")
    # ps.save(model_abs_path)

    print(f"Saving losses in {losses_abs_path}...")
    with open(losses_abs_path, "w") as f:
        f.write("\n".join([str(l) for l in losses]))

    print(f"Saving times taken to {traintime_abs_path}...")
    with open(traintime_abs_path, 'w', encoding='utf-8') as f:
        f.write("Configuration in: " + args.config_file + "\n")
        f.write(str(cfg))

        take_time("saving results")
        take_time.total()
        f.write("\n\nTimes taken:\n" + str(take_time))

    print("\nTimes taken:\n", take_time)



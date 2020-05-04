"""
This script performs training of ParagraphSelector models.
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

    model_abs_path = cfg('model_abs_dir') + args.model_name + "/"
    #model_abs_path += '.pt' if not args.model_name.endswith('.pt') else ''
    losses_abs_path = model_abs_path + args.model_name + ".losses"
    devscores_abs_path = model_abs_path + args.model_name + ".devscores"
    traintime_abs_path = model_abs_path + args.model_name + ".times"

    # check all relevant file paths and directories before starting training
    for path in [cfg("data_abs_path"), cfg("dev_data_abs_path")]:
        try:
            f = open(path, "r")
            f.close()
        except FileNotFoundError as e:
            print(e)
            sys.exit()
    for path in [model_abs_path]:
        if not os.path.exists(path):
            print(f"newly creating {path}")
            os.makedirs(path)

    take_time("parameter input")


    # ========== DATA PREPARATION
    # TODO add comments
    # TODO improve variable/parameter names

    # try to load pickled data, and in case it doesn't work, read the whole HotPotQA dataset
    try:
        with open(cfg("pickled_train_data"), "rb") as f:
            train_data_raw = pickle.load(f)
            dataset_size = cfg("dataset_size") if cfg("dataset_size") else len(train_data_raw)
        with open(cfg("pickled_dev_data"), "rb") as f:
            dev_data_raw = pickle.load(f)
            # restrict loaded dev data to the required percentage
            dev_data_raw = dev_data_raw[:int(cfg('percent_for_eval_during_training') * dataset_size)]

    except:  # TODO why does it always go to this exception instead of loading the pickled data?
        print(f"Reading data from {cfg('data_abs_path')}...")  # the whole HotPotQA training set
        dh = HotPotDataHandler(cfg("data_abs_path"))
        data = dh.data_for_paragraph_selector()  # get raw points
        dataset_size = cfg("dataset_size") if cfg("dataset_size") else len(data)

        print("Splitting data...")  # split into what we need DURING training
        train_data_raw, dev_data_raw = train_test_split(data[:dataset_size],
                                                        # restricts the number of training+dev examples
                                                        test_size=cfg('percent_for_eval_during_training'),
                                                        random_state=cfg('shuffle_seed'),
                                                        shuffle=True)

        if cfg("pickled_train_data") and cfg("pickled_dev_data"):
            print(f"Pickling train/dev data for later re-use.")
            print(f"Destinations: \n"
                  f"   {cfg('pickled_train_data')}\n"
                  f"   {cfg('pickled_dev_data')}")
            with open(cfg("pickled_train_data"), "wb") as f:
                pickle.dump(train_data_raw, f)

            with open(cfg("pickled_dev_data"), "wb") as f:
                pickle.dump(dev_data_raw, f)

    # ParagraphSelector.train() requires this step
    train_data = ParagraphSelector.make_training_data(train_data_raw,
                                                      text_length=cfg("text_length"))
    # group training data into batches
    #bs = cfg("batch_size")
    #N = len(train_data)
    #train_data = [train_data[i: i + bs] for i in range(0, N, bs)]

    take_time("data preparation")


    #========== TRAINING
    print("Initialising ParagraphSelector...")
    ps = ParagraphSelector.ParagraphSelector(cfg("bert_model_path"))

    print(f"training for {cfg('epochs')} epochs...")
    losses, dev_scores = ps.train(train_data, # pre-processed data as tensors
                          dev_data_raw, # lightly processed data; lists, strings etc.
                          model_abs_path,
                          epochs=cfg("epochs"),
                          batch_size=cfg("batch_size"),
                          learning_rate=cfg("learning_rate"),
                          eval_interval=cfg("eval_interval"),
                          try_gpu=cfg("try_gpu"))
    take_time(f"training")

    # print(f"Saving model in {model_abs_path}...")
    # ps.save(model_abs_path)

    print(f"Saving losses in {losses_abs_path}...")
    with open(losses_abs_path, "w") as f:
        f.write("\n".join([str(l) for l in losses]))

    print(f"Saving dev scores in {devscores_abs_path}...")
    with open(devscores_abs_path, "w") as f:
        score_strings = []
        for tup in dev_scores:
            score_strings.append("\t".join([str(v) for v in tup]))
        f.write("\n".join(score_strings))

    print(f"Saving times taken to {traintime_abs_path}...")
    with open(traintime_abs_path, 'w', encoding='utf-8') as f:
        f.write("Configuration in: " + args.config_file + "\n")
        f.write(str(cfg))

        take_time("saving results")
        take_time.total()
        f.write("\n\nTimes taken:\n" + str(take_time))

    print("\nTimes taken:\n", take_time)



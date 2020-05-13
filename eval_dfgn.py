""" This script is for evaluation of a DFGN model (including Paragraph Selector) """



#TODO
# - input config
# - load data
# - load PS ps = torch.load()
# - load DFGN = torch.load()
# - if we want to exclude PS errors: call make_eval_data and evaluate on the resulting data
# - call train_dfgn.evaluate()
# - output results


import torch

from utils import Timer
from utils import HotPotDataHandler
from utils import ConfigReader

from modules import ParagraphSelector
from train_dfgn import DFGN

import argparse
import sys
import os



import os, sys, argparse
import pickle # mainly for training data
import torch
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

import hotpot_evaluate_v1 as official_eval_script
import flair # for NER in the EntityGraph
from modules import ParagraphSelector, EntityGraph, Encoder, FusionBlock, Predictor
from train_dfgn import predict
import utils




def prepare_prediction(raw_data_points,
                       para_selector, ps_threshold, text_length,
                       ner_tagger,
                       timer):
    """
    #TODO docstring
    :param raw_data_points: list of raw_points or just a single raw_point
    :param para_selector:
    :param ps_threshold:
    :param text_length:
    :param ner_tagger:
    :return: required data if possible, or None if data not usable by the network's components
    """

    graph_log = [0,0,0] # [total nodes, total connections, number of graphs]
    point_usage_log = [0,0] # [used points, unused points]

    ids = []
    queries = []
    contexts = []
    graphs = []
    sent_lengths = []

    raw_data_points = [raw_data_points] if type(raw_data_points)!=list else raw_data_points
    for i, point in enumerate(raw_data_points):

        """ DATA PROCESSING """
        # make a list[ list[str, list[str]] ] for each point in the batch
        context = para_selector.make_context(point,
                                             threshold=ps_threshold,
                                             context_length=text_length)
        timer.again("ParagraphSelector_prediction")
        graph = EntityGraph.EntityGraph(context,
                                        context_length=text_length,
                                        tagger=ner_tagger)
        timer.again("EntityGraph_construction")

        if graph.graph:
            ids.append(point[0])
            queries.append(point[2])
            contexts.append(context)
            graphs.append(graph)
            sent_lengths.append(utils.sentence_lengths(context, tokenizer))
            graph_log = [a + b  # [total nodes, total connections, number of graphs]
                         for a, b in
                         zip(graph_log, [len(graph.graph),
                                         len(graph.relation_triplets()),
                                         1])]
            point_usage_log[0] += 1
        else:  # if the NER in EntityGraph doesn't find entities, the datapoint is useless.
            point_usage_log[1] += 1

    # update the batch to exclude useless data points
    if not ids:
        print("In eval_dfgn.prepare_prediction(): No usable data points; return value: None")
        return None, None, None, None, None, \
               timer, graph_log, point_usage_log
    else:
        return ids, queries, contexts, graphs, sent_lengths,\
               timer, graph_log, point_usage_log


def encode_to_device(queries, contexts, graphs, encoder, device, timer):
    """
    #TODO docstring
    :param queries:
    :param contexts:
    :param encoder:
    :param device:
    :return: lists of query/context token id tensors, graphs, timer (all tensor parts moved to the device)
    """

    # turn the texts into tensors in order to put them on the GPU
    qc_ids = [encoder.token_ids(q, c) for q, c in zip(queries, contexts)]  # list[ (list[int], list[int]) ]
    q_ids, c_ids = list(zip(*qc_ids))  # tuple(list[int]), tuple(list[int])
    q_ids_list = [torch.tensor(q).to(device) for q in q_ids]  # list[Tensor]
    c_ids_list = [torch.tensor(c).to(device) for c in c_ids]  # list[Tensor]

    for i, g in enumerate(graphs):
        graphs[i].M = g.M.to(device)  # work with enumerate to actually mutate the graph objects

    timer.again("encode_to_device")

    return q_ids_list, c_ids_list, graphs, timer



# =========== PARAMETER INPUT
take_time = Timer()

parser = argparse.ArgumentParser()
parser.add_argument('config_file', metavar='config', type=str,
                    help='configuration file for evaluation')
parser.add_argument('dfgn_model_name', metavar='model', type=str,
                    help="name of the DFGN model's file")
args = parser.parse_args()
cfg = ConfigReader(args.config_file)

model_abs_path = cfg('model_abs_dir') + args.dfgn_model_name + "/" # there is a .bin in this directory; that's the model
results_abs_path = model_abs_path + args.dfgn_model_name + ".test_scores"
predictions_abs_path = cfg('predictions_abs_dir') + args.dfgn_model_name + ".predicitons"

# check all relevant file paths and directories before starting training
try:
    f = open(cfg("test_data_abs_path"), "r")
    f.close()
except FileNotFoundError as e:
    print(e)
    sys.exit()


if not os.path.exists(model_abs_path):
    print(f"newly creating {model_abs_path}")
    os.makedirs(model_abs_path)

if not os.path.exists(cfg('predictions_abs_dir')):
    print(f"newly creating {cfg('predictions_abs_dir')}")
    os.makedirs(cfg('predictions_abs_dir'))
else:
    print(f"overwriting {predictions_abs_path} with new predictions!")
    with open(predictions_abs_path, 'w') as f:  #
        f.write("")

# handle GPU usage (all parts on the same device)
device = torch.device('cpu')
if cfg("try_gpu") and torch.cuda.is_available():
    torch.cuda.set_device(cfg("gpu_number") if cfg("gpu_number") else 0)
    device = torch.device('cuda')

take_time("parameter input")


# =========== DATA LOADING
print(f"Reading data from {cfg('test_data_abs_path')}...")  # the whole HotPotQA training set
dh = HotPotDataHandler(cfg("test_data_abs_path")) # the whole HotPotQA dev set
raw_data = dh.data_for_paragraph_selector() # get raw points
data_limit = cfg("testset_size") if cfg("testset_size") else len(raw_data)
take_time("data loading")


# =========== MODEL LOADING
# load all the nerual models involved
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # needed?

flair.device = device
ner_tagger = flair.models.SequenceTagger.load('ner')  # this hard-codes flair tagging!

para_selector = ParagraphSelector.ParagraphSelector(cfg("ps_model_abs_dir")) # looks for the 'pytorch_model.bin' in this directory
para_selector.net.eval() # ParagraphSelector itself does not inherit from nn.Module.
para_selector = para_selector.net.to(device)

dfgn = torch.load(model_abs_path)
dfgn.eval()
dfgn = dfgn.to(device)
take_time("model loading")


# =========== PREDICTIONS
counter = 0 # counts up with each question ( = each data point)

batch_size = data_limit if not cfg(["prediction_batch_size"]) else cfg(["prediction_batch_size"])
for pos in range(0, data_limit, batch_size):

    answers = [] # predicted answers
    sp = []      # predicted supporting facts

    # prepare data: select paragraphs, make graphs, ...
    # shape of sent_lengths: list[ list[list[int]] ] sentences' lengths per paragraph; for multiple data points
    ids, queries, contexts, graphs, sent_lengths, \
    take_time, graph_log, point_usage_log = prepare_prediction(raw_data[pos : pos+data_limit],
                                                               para_selector,
                                                               cfg("ps_threshold"),
                                                               cfg("text_length"),
                                                               ner_tagger,
                                                               take_time)
    # encode strings to IDs and put the tensors on the device
    queries, contexts, graphs, take_time = encode_to_device(queries,
                                                            contexts,
                                                            graphs,
                                                            dfgn.encoder,
                                                            device,
                                                            take_time)

    for id, query, context, graph, s_lens in tqdm(zip(ids, queries, contexts, graphs, sent_lengths),desc="predictions"):
        counter += 1 # just for keeping track.
        if cfg("verbose_evaluation"): print(f"({counter}) {id}\n   {query}")

        answer, sup_fact_pairs = predict(dfgn, query, context, graph,
                                         tokenizer, s_lens, fb_passes=cfg("fb_passes"))
        take_time.again("prediction")

        answers[id] = answer  # {question_id: str}
        sp[id] = sup_fact_pairs  # {question_id: list[list[paragraph_title, sent_num]]}
        if cfg("verbose_evaluation"): print(f"   {answer}\n")

    with open(predictions_abs_path, 'a') as f:
        json.dump({"answer": answers, "sp": sp}, f)
    take_time.again("dump_predictions")



#=========== EVALUATION
metrics = official_eval_script.eval(predictions_abs_path, cfg("test_data_abs_path"))

o_sup, o_start, o_end, o_type = net(query, context, graph,
                                        fb_passes=fb_passes)  # (M, 2), (M), (M), (1, 3)
    sups.append(o_sup)
    starts.append(o_start)
    ends.append(o_end)
    types.append(o_type)

sups = torch.stack(sups)  # (batch, M, 2)
starts = torch.stack(starts)  # (batch, 1, M)
ends = torch.stack(ends)  # (batch, 1, M)
types = torch.stack(types)  # (batch, 1, 3)




THE FOLLOWING IS FROM eval_ps.py
print("Evaluating...")
precision, recall, f1, accuracy, ids, y_true, y_pred = model.evaluate(raw_data[:data_limit],
                                                            threshold=cfg("threshold"),
                                                            text_length=cfg("text_length"),
                                                            try_gpu=cfg("try_gpu"))
print("Precision:", precision)
print("Recall:   ", recall)
print("F score:  ", f1)
print("Accuracy: ", accuracy)
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
            "\nF score:   " + str(f1) +
            "\nAccuracy:  " + str(accuracy) + "\n")
    f.write("Hyper parameters:\n" + str(cfg))

    take_time.total()
    f.write("\n\nTimes taken:\n" + str(take_time))
    print("\ntimes taken:\n", take_time)





THE FOLLOWING IS FROM train_dfgn.py
# ========= END OF TRAINING =============#
    metrics = evaluate(net,  # TODO make this prettier
                       tokenizer, ner_tagger,
                       training_device, dev_data_filepath, dev_preds_filepath,
                       fb_passes=fb_passes,
                       text_length=text_length,
                       verbose=verbose_evaluation)
    score = metrics["joint_f1"]
    dev_scores.append(metrics)  # appends the whole dict of metrics
    if score >= best_score:
        torch.save(net,
                   model_save_path)

    if not a_model_was_saved_at_some_point:  # make sure that there is a model file
        print(f"saving model to {model_save_path}...")
        torch.save(net, model_save_path)

# ========== LOGGING
print(f"Saving losses in {losses_abs_path}...")
with open(losses_abs_path, "w") as f:
    f.write("batch_size\toverall_loss\tsup_loss\tstart_loss\tend_loss\ttype_loss\n")
    f.write("\n".join(["\t".join([str(l) for l in step]) for step in losses]))

print(f"Saving dev scores in {devscores_abs_path}...")
with open(devscores_abs_path, "w") as f:
    for metrics_dict in dev_scores:
        f.write(str(metrics_dict))  # this can be parsed with ast.literal_eval() later on

print(f"Saving config and times taken to {traintime_abs_path}...")
with open(traintime_abs_path, 'w', encoding='utf-8') as f:
    f.write("Configuration in: " + args.config_file + "\n")
    f.write(str(cfg) + "\n")

    f.write("\n Times taken per step:\n" + str(train_times) + "\n")
    take_time("saving results")
    take_time.total()
    f.write("\n Overall times taken:\n" + str(take_time) + "\n")

    # graph_logging = [total nodes, total connections, number of graphs]
    f.write("\nGraph statistics:\n")
    f.write("connections per node:       " + str(graph_logging[1] / float(graph_logging[0])) + "\n")
    f.write("nodes per graph (limit:40): " + str(graph_logging[0] / float(graph_logging[2])) + "\n")

    # point_usage = [used points, unused points]
    f.write("\nData usage statistics:\n")
    f.write("Overall points:       " + str(sum(point_usage)) + "\n")
    f.write("used/unused points:   " + str(point_usage[0]) + " / " + str(point_usage[1]) + "\n")
    f.write("Ratio of used points: " + str(point_usage[0] / float(sum(point_usage))) + "\n")

print("\nTimes taken:\n", take_time)
print("done.")
""" This script is for evaluation of a DFGN model (including Paragraph Selector) """

import os, sys, argparse
import json
import torch
from tqdm import tqdm

import flair # for NER in the EntityGraph
from transformers import BertTokenizer
import hotpot_evaluate_v1 as official_eval_script

import utils
from utils import Timer
from utils import HotPotDataHandler
from utils import ConfigReader
from modules import ParagraphSelector, EntityGraph
from train_dfgn import predict, DFGN


def prepare_prediction(raw_data_points,
                       para_selector, ps_threshold, text_length,
                       ner_tagger,
                       timer):
    """
    Starting from a raw point (or a list of raw points), prepare all
    the data structures required by the DFGN module in order to predict
    an output.

    :param raw_data_points: list of raw_points or just a single raw_point
    :param para_selector: a ParagraphSelector object
    :param ps_threshold: threshold for the paragraph selector (relevance score between paragraph and query)
    :param text_length: max text length for the context
    :param ner_tagger: NER tagger
    :param timer: a timer object (see utils)
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
    :param graphs:
    :param encoder:
    :param device:
    :param timer: a timer object (see utils)
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

def output_scores(metrics, mode='print', filehandle=None):
    """
    Print results to screen or write them to a file.

    :param metrics: a metrics dictionary as returned by the official HotPotQA evaluation script (hotpot_evaluate_v1)
    :param mode: string - either 'print' or 'write'
    :param filepath: a filepath where the results should be written if in write mode
    """
    if mode == 'print':
        print("=========================")
        print("ANSWER SCORES")
        print("Precision:  ", metrics["prec"])
        print("Recall:     ", metrics["recall"])
        print("F score:    ", metrics["f1"])
        print("exact match:", metrics["em"])
        print('-------------------------\n')
        print("SUPPORTING FACT SCORES")
        print("Precision:  ", metrics["sp_prec"])
        print("Recall:     ", metrics["sp_recall"])
        print("F score:    ", metrics["sp_f1"])
        print("exact match:", metrics["sp_em"])
        print('-------------------------\n')
        print("SUPPORTING FACT SCORES")
        print("Precision:  ", metrics["joint_prec"])
        print("Recall:     ", metrics["joint_recall"])
        print("F score:    ", metrics["joint_f1"])
        print("exact match:", metrics["joint_em"])
        print('=========================\n')
    elif mode == 'write' and filehandle:
        f.write("\n=========================")
        f.write("\nANSWER SCORES")
        f.write("\nPrecision:  " + str(metrics["prec"]))
        f.write("\nRecall:     " + str(metrics["recall"]))
        f.write("\nF score:    " + str(metrics["f1"]))
        f.write("\nexact match:" + str(metrics["em"]))
        f.write('\n-------------------------\n')
        f.write("\nSUPPORTING FACT SCORES")
        f.write("\nPrecision:  " + str(metrics["sp_prec"]))
        f.write("\nRecall:     " + str(metrics["sp_recall"]))
        f.write("\nF score:    " + str(metrics["sp_f1"]))
        f.write("\nexact match:" + str(metrics["sp_em"]))
        f.write('\n-------------------------\n')
        f.write("\nSUPPORTING FACT SCORES")
        f.write("\nPrecision:  " + str(metrics["joint_prec"]))
        f.write("\nRecall:     " + str(metrics["joint_recall"]))
        f.write("\nF score:    " + str(metrics["joint_f1"]))
        f.write("\nexact match:" + str(metrics["joint_em"]))
        f.write('\n=========================\n')
    else:
        print("WARNING: could neither print, nor write the scores!",
              "Check your file paths!")

def write_results(filename, metrics, timer, graph_stats, point_stats):
    """
    Write results and stats to file to a file.

    :param filename: path to file where output should be written
    :param metrics: a metrics dictionary as returned by the official HotPotQA evaluation script (hotpot_evaluate_v1)
    :param timer: a timer object (utils) used during evaluation
    :param graph_stats: stats about the EntityGraph instance used,
                        list[int, int, int] - [total nodes, total connections, number of graphs]
    :param point_stats: stats about number of used points (unused points are those with no graph connections)
                        list[int, int] - [used points, unused points]
    :return: a timer object (utils)
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Configuration in: " + args.config_file + "\n")
        f.write("Outputs in:  " + predictions_abs_path + "\n")
        f.write("\n")
        f.write("Evaluation parameters:\n" + str(cfg) + "\n")
        f.write("\n")
        output_scores(metrics, mode='write', filehandle=f)

        timer.total()
        f.write("\n\nTIMES TAKEN:\n" + str(timer))
        print("\ntimes taken:\n", timer)

        # graph_logging = [total nodes, total connections, number of graphs]
        f.write("\nGRAPH STATISTICS:\n")
        f.write("connections per node:       " + str(graph_stats[1] / float(graph_stats[0])) + "\n")
        f.write("nodes per graph (limit:40): " + str(graph_stats[0] / float(graph_stats[2])) + "\n")

        # point_usage = [used points, unused points]
        f.write("\nDATA USAGE STATISTICS:\n")
        f.write("Overall points:       " + str(sum(point_stats)) + "\n")
        f.write("used/unused points:   " + str(point_stats[0]) + " / " + str(point_stats[1]) + "\n")
        f.write("Ratio of used points: " + str(point_stats[0] / float(sum(point_stats))) + "\n")

        return timer

# =========== PARAMETER INPUT
take_time = Timer()

parser = argparse.ArgumentParser()
parser.add_argument('config_file', metavar='config', type=str,
                    help='configuration file for evaluation')
parser.add_argument('dfgn_model_name', metavar='model', type=str,
                    help="name of the DFGN model's file")
args = parser.parse_args()
cfg = ConfigReader(args.config_file)

model_abs_dir = cfg('model_abs_dir') + args.dfgn_model_name + "/" # there is a .bin in this directory; that's the model
results_abs_path = model_abs_dir + args.dfgn_model_name + ".test_scores"
predictions_abs_path = cfg('predictions_abs_dir') + args.dfgn_model_name + ".predicitons"

# check all relevant file paths and directories before starting training
try:
    f = open(cfg("test_data_abs_path"), "r")
    f.close()
except FileNotFoundError as e:
    print(e)
    sys.exit()


if not os.path.exists(model_abs_dir):
    print(f"newly creating {model_abs_dir}")
    os.makedirs(model_abs_dir)

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
para_selector.net = para_selector.net.to(device)

dfgn = torch.load(model_abs_dir+args.dfgn_model_name)
dfgn.eval()
dfgn = dfgn.to(device)
take_time("model loading")


# =========== PREDICTIONS
counter = 0 # counts up with each question ( = each data point)
graph_stats = []
point_usage_stats = []

batch_size = data_limit if not cfg("prediction_batch_size") else cfg("prediction_batch_size")
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

    graph_stats = [old+new for old,new in zip(graph_stats, graph_log)]
    point_usage_stats = [old+new for old,new in zip(point_usage_stats, point_usage_log)]



#=========== EVALUATION
print("Evaluating...")
metrics = official_eval_script.eval(predictions_abs_path, cfg("test_data_abs_path"))
#{'em',       'f1',       'prec',       'recall',
# 'sp_em',    'sp_f1',    'sp_prec',    'sp_recall',
# 'joint_em', 'joint_f1', 'joint_prec', 'joint_recall'}

output_scores(metrics, mode='print')
take_time("evaluation")

#========== LOGGING
take_time = write_results(results_abs_path, metrics, take_time, graph_stats, point_usage_stats)
print("\nTimes taken:\n", take_time)
print("done.")
#!/usr/bin/python3.6
#author: Simon Preissner

"""
This module provides helper functions.
"""

import os
import sys
import re
import json
#from tqdm import tqdm
from time import time
from torch import nn
import torch
from torch import functional as F
from torch.nn import functional as nnF
import string
import difflib

from pprint import pprint

def loop_input(rtype=str, default=None, msg=""):
    """
    Wrapper function for command-line input that specifies an input type
    and a default value. Input types can be string, int, float, or bool,
    or "file", so that only existing files will pass the input.
    :param rtype: type of the input. one of str, int, float, bool, "file"
    :type rtype: type
    :param default: value to be returned if the input is empty
    :param msg: message that is printed as prompt
    :type msg: str
    :return: value of the specified type
    """
    while True:
        try:
            s = input(msg+f" (default: {default}): ")
            if rtype == bool and len(s) > 0:
                if s=="True":
                    return True
                elif s=="False":
                    return False
                else:
                    print("Input needs to be convertable to",rtype,"-- try again.")
                    continue
            if rtype == "filepath":
                s = default if len(s) == 0 else s
                try:
                    f = open(s, "r")
                    f.close()
                    return s
                except FileNotFoundError as e:
                    print("File",s,"not found -- try again.")
                    continue
            else:
                return rtype(s) if len(s) > 0 else default
        except ValueError:
            print("Input needs to be convertable to",rtype,"-- try again.")
            continue

def flatten_context(context, siyana_wants_a_oneliner=False):
        """
        return the context as a single string,
        :param context: list[ list[ str, list[str] ] ]
        :return: string containing the whole context
        """

        if siyana_wants_a_oneliner:  # This is for you, Siyana!
            return " ".join([p[0] + " " + " ".join(["".join(s) for s in p[1:]]) for p in context])

        final = ""
        for para in context:
            for sent in para:
                if type(sent) == list:
                    final += "".join(sent) + " "
                else:
                    final += sent + " "
        final = final.rstrip()
        return final

class ConfigReader():
    """
    Basic container and management of parameter configurations.
    Read a config file (typically ending with .cfg), use this as container for
    the parameters during runtime, and change/write parameters.

    CONFIGURATION FILE SYNTAX
    - one parameter per line, containing a name and a value
        - name and value are separated by at least one white space or tab
        - names should contain alphanumeric symbols and '_' (no '-', please!)
    - list-like values are allowed (use Python list syntax)
        - strings within value lists don't need to be quoted
        - value lists either with or without quotation (no ["foo", 3, "bar"] )
        - mixed lists will exclude non-quoted elements
    - multi-word expressions are marked with single or double quotation marks
    - TODO strings containing quotation marks are not tested yet. Be careful!
    - lines starting with '#' are ignored
    - no in-line comments!
    - config files should have the extension 'cfg' (to indicate their purpose)
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.params = self.read_config()

    def __repr__(self):
        """
        returns tab-separated key-value pairs (one pair per line)
        """
        return "\n".join([str(k)+"\t"+str(v) for k,v in self.params.items()])

    def __call__(self, *paramnames):
        """
        Returns a single value or a list of values corresponding to the
        provided parameter name(s). Returns the whole config in form of a
        dictionary if no parameter names are specified.
        """
        if not paramnames: # return the whole config
            return self.params
        else: # return specified values
            values = []
            for n in paramnames:
                if n in self.params:
                    values.append(self.params[n])
                else:
                    print(f"WARNING: couldn't find parameter {n}.")
                    print(f"   Make sure to include it in {self.filepath}")
                    print(f"   Continuing with value {None} for {n}")
                    values.append(None)
            return values[0] if len(values) == 1 else values

    def read_config(self):
        """
        Reads the ConfigReader's assigned file (attribute: 'filename') and parses
        the contents into a dictionary.
        - ignores empty lines and lines starting with '#'
        - takes the first continuous string as parameter key (or: parameter name)
        - parses all subsequent strings (splits at whitespaces) as values
        - tries to convert each value to float, int, and bool. Else: string.
        - parses strings that look like Python lists to lists
        :return: dict[str:obj]
        """
        cfg = {}
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip()
            if not line: # ignore empty lines
                continue
            elif line.startswith('#'): # ignore comment lines
                continue

            words = line.split()
            paramname = words.pop(0)
            if not words: # no value specified
                print(f"WARNING: no value specified for parameter {paramname}.")
                paramvalue = None
            elif words[0].startswith("["): # detects a list of values
                paramvalue = self.listparse(" ".join(words))
            elif words[0].startswith('"') or words[0].startswith('\''): # detects a multi-word string
                paramvalue = self.stringparse(words)
            else:
                """ only single values are valid! """
                if len(words) > 1:
                    #TODO make this proper error handling (= throw an error)
                    print(f"ERROR while parsing {self.filepath} --",
                          f"too many values in line '{line}'.")
                    sys.exit()
                else:
                    """ parse the single value """
                    paramvalue = self.numberparse(words[0]) # 'words' is still a list
                    paramvalue = self.boolparse(paramvalue)

            cfg[paramname] = paramvalue # adds the parameter to the config

        self.config = cfg
        return self.config

    @classmethod
    def listparse(cls, liststring):
        """
        Parses a string that looks like a Python list (square brackets, comma
        separated, ...). A list of strings can make use of quotation marks, but
        doesn't need to. List-like strings that contain some quoted and some
        unquoted elements will be parsed to only return the quoted elements.
        Elements parsed from an unquoted list will be converted to numbers/bools
        if possible.
        Examples:
            [this, is, a, valid, list] --> ['this', 'is', 'a', 'valid', 'list']
            ["this", "one", "too"]     --> ['this', 'one', 'too']
            ['single', 'quotes', 'are', 'valid'] --> ['single', 'quotes', 'are', 'valid']
            ["mixing", 42, is, 'bad']  --> ['mixing', 'bad']
            ["54", "74", "90", "2014"] --> ['54', '74', '90', '2014']
            [54, 74, 90, 2014]         --> [54, 74, 90, 2014]
            [True, 1337, False, 666]   --> [True, 1337, False, 666]
            [True, 1337, "bla", False, 666] --> ['bla']
        """
        re_quoted = re.compile('["\'](.+?)["\'][,\]]')
        elements = re.findall(re_quoted, liststring)
        if elements:
            return elements

        re_unquoted = re.compile('[\[\s]*(.+?)[,\]]')
        elements = re.findall(re_unquoted, liststring)
        if elements:
            result = []
            for e in elements:
                e = cls.numberparse(e)  # convert to number if possible
                e = cls.boolparse(e)  # convert to bool if possible
                result.append(e)
            return result

    @staticmethod
    def stringparse(words):
        words[0] = words[0][1:] # delete opening quotation marks
        words[-1] = words[-1][:-1] #delete closing quotation marks
        return " ".join(words)

    @staticmethod
    def numberparse(string):
        """
        Tries to convert 'string' to a float or even int.
        Returns int/float if successful, or else the input string.
        """
        try:
            floaty = float(string)
            if int(floaty) == floaty:
                return int(floaty)
            else:
                return floaty
        except ValueError:
            return string

    @staticmethod
    def boolparse(string):
        if string == "True":
            return True
        elif string == "False":
            return False
        else:
            return string

    def get_param_names(self):
        """
        returns a list of parameter names
        """
        return [key for key in self.params.keys()]

    def set(self, paramname, value):
        self.params.update({paramname:value})

class Timer():
    """
    Simple wrapper for taking time. A Timer object starts taking time upon initialization.
    Take time by calling the object with a description of the timed action, e.g.:
    my_timer('preprocessing').
    Take times of recurring activities with again().
    Take the overall time with total().
    The string representation outputs all times taken, or only certain times, if specified.
    """
    def __init__(self):
        self.T0 = time()
        self.t0 = time()
        self.times = {}
        self.steps = []
        self.period_name = ""

    def __call__(self, periodname):
        span = time() - self.t0
        self.t0 = time()
        self.steps.append(periodname)
        self.times.update({periodname: span})
        return span

    def __repr__(self, *args):
        steps = [s for s in args if s in self.steps] if args else self.steps
        return "\n".join([str(round(self.times[k], 5)) + "   " + k for k in steps])

    def again(self, periodname):
        """
        Take cumulative time of a recurring activity.
        :param periodname: str -- description of the activity
        :return: float -- time in seconds taken for current iteration of the activity
        """
        span = time() - self.t0
        self.t0 = time()
        if periodname in self.times:
            self.times[periodname] += span
        else:
            self.steps.append(periodname)
            self.times[periodname] = span
        return span

    def total(self):
        span = time() - self.T0 # with a capital T, to take the overall time
        self.steps.append("total")
        self.times.update({"total": span})
        return span

class HotPotDataHandler():
    """
    This class provides an interface to the HotPotQA dataset.
    It loads data and extracts the required information.
    """

    def __init__(self, filename="./data/hotpot_train_v1.1.json"):
        self.filename = os.path.abspath(filename)
        with open(self.filename, "r") as f:
            self.data = json.load(f)

    def __repr__(self):
        header = str(len(self.data))+" items in data; keys:\n"
        content = "\n".join([str(k) for k in self.data[0].keys()]).rstrip()
        return header+content

    def data_for_paragraph_selector(self):
        """
        This method makes what is called "raw_point" in other parts of the project.

        From the HotPotQA data, extract the required information and return it
        as a list of tuples (question_id, supporting_facts, query, paragraphs, answer).
        Shapes of the tuples' elements:
        - question_id: str
        - supporting_facts: set[str]
        - query: str
        - paragraphs: 10-element list[str, list[str]]
            - first element: paragraph title
            - second element: list of the paragraph's sentences
        - answer: str
        - supp_facts_detailed: dict{str: [int]}

        :return: list(tuple( str, list[str], str, list[str,list[str]], str ))
        """
        result = []
        for point in self.data:
            # supp_facts = set([fact[0] for fact in point["supporting_facts"]])

            supp_facts_detailed = {}
            for fact in point["supporting_facts"]:
                if supp_facts_detailed.get(fact[0]):
                    supp_facts_detailed[fact[0]].append(fact[1])
                else:
                    supp_facts_detailed[fact[0]] = [fact[1]]
            result.append(tuple((point["_id"],
                                 supp_facts_detailed, # we used to use supp_facts here
                                 point["question"],
                                 point["context"],
                                 point["answer"])))
        return result

def make_labeled_data_for_predictor(graph, raw_point, tokenizer):
    """
    Prepare labeled data for the Predictor, i.e. per-token labels for
    1. is_supporting_fact
    2. is_answer_start
    3. is_answer_end
    4. question type (not per-token; one label per datapoint)

    From the graph we get:
        - a list of tokens
        - the context (titles + sentences)
    From the raw_point we get:
        - supporting facts
        - answer

    :param graph: instance of the EntityGraph class (holds context with M tokens)
    :param raw_point: data point as returned from HotPotDataHandler.data_for_paragraph_selector()
    :return sup_labels: Tensor of shape M -- marks tokens that are 'supporting facts'
    :return start_labels: Tensor of shape M -- marks tokens which are start of spans
    :return end_labels: Tensor of shape M -- marks tokens which are end of spans
    :return type_labels: Tensor of shape 1 -- one of 3 question types (yes/no/span)
    :return sup_labels_by_sentence: #TODO describe
    :return sentence_lengths: #TODO describe
    """
    M = len(graph.tokens)

    sup_labels = torch.zeros(M, dtype=torch.long)
    start_labels = torch.zeros(M, dtype=torch.long)
    end_labels = torch.zeros(M, dtype=torch.long)
    type_labels = torch.zeros(1, dtype=torch.long)

    answer = raw_point[4].lower()

    # get answer type
    if answer == "yes":
        type_labels[0] = 0
    elif answer == "no":
        type_labels[0] = 1
    else:
        type_labels[0] = 2

    # if the answer is not "yes" or "no", its a span
    if type_labels[0] == 2:
        for i, token in enumerate(graph.tokens):
            if answer.startswith(token):
                start_labels[i]=1
            if answer.endswith(token):
                end_labels[i]=1

    # get supporting facts (paragraphs)
    list_context = [[p[0] + " "] + p[1] for p in graph.context]  # squeeze header into the paragraph
    num_sentences = sum([len(p) for p in list_context]) # number of sentence, including headers

    sup_labels_by_sentence = torch.zeros(num_sentences, dtype = torch.long) #TODO needed? originally intended for custom evaluation (we probably use the official eval script)

    # use an extra tokenizer again in order to have the correct number of tokens in order to determine position later
    tokenized_sentences = [[tokenizer.tokenize(s) for s in p] for p in list_context] # list[list[list[str]]]
    sentence_lengths = [[len(s) for s in p] for p in tokenized_sentences] # list[list[int]]

    position = 0
    sent_position = 0
    for i, para in enumerate(graph.context):
        if raw_point[1].get(para[0]): # para title in supporting facts
            for j, sent in enumerate(tokenized_sentences[i]):
                if j - 1 in raw_point[1][para[0]]: # the 0th sentence is the paragraph title, j - 1 accounts for that
                    sup_labels[ position : position + len(sent) ] = 1 # fill with 1's from position to position + len(sent)
                    sup_labels_by_sentence[sent_position] = 1
                position += len(sent) # update position
                sent_position += 1
        else: # if the paragraph does not have any supporting facts, update our position with the total paragraph length
            position += sum([len(sent) for sent in tokenized_sentences[i]])
            sent_position += len(tokenized_sentences[i])

    return (sup_labels, start_labels, end_labels, type_labels, sup_labels_by_sentence, sentence_lengths) # M, M, M, 1

def make_eval_data(raw_points):
    """
    TODO: docstring
    :param raw_points:
    :return:
    """

    eval_data = {"answer": {},
                 "sp": {}}
    for point in raw_points:
        eval_data["answer"][point[0]] = point[4] # map question IDs to answers
        # map question IDs to supporting facts
        eval_data["sp"][point[0]] = []

        for para in point[3]: # point
            # if the paragraph has supporting facts;
            if point[1].get(para[0]): # point[1] is {str:list[int]}; para[0] is the paragraph title
                for sent_idx in point[1][para[0]]: # for each sentence that's a supporting fact
                    # if the sentence with index sent_idx is in the paragraph (has not been cropped out by PS)
                    if sent_idx < len(para[1]): # para[1] is a list of sentences in the paragraph
                        eval_data["sp"][point[0]].append([para[0], sent_idx])

    return eval_data


class Linear(nn.Module):
    '''
    Taken from Taeuk Kim's re-implementation of BiDAF:
    https://github.com/galsang/BiDAF-pytorch/blob/master/utils/nn.py
    This class is used for all layers in the BiDAF architecture.
    '''
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x

class BiDAFNet(torch.nn.Module):
    """
    This class is an adaptation from a pytorch implementation of the
    Bi-Directional Attention Flow (BiDAF) architecture described in
    Seo et al. (2016) [https://arxiv.org/abs/1611.01603].
    The present code is in most parts copied from model.BiDAF.att_flow_layer()
    in Taeuk Kim's re-implementation: https://github.com/galsang/BiDAF-pytorch ,
    and slightly adapted.
    """

    def __init__(self, hidden_size=768, output_size=300, dropout=0.0):
        super(BiDAFNet, self).__init__()

        self.att_weight_c = Linear(hidden_size, 1, dropout=dropout)
        self.att_weight_q = Linear(hidden_size, 1, dropout=dropout)
        self.att_weight_cq = Linear(hidden_size, 1, dropout=dropout)

        self.reduction_layer = Linear(hidden_size * 4, output_size, dropout=dropout)

    def forward(self, emb1, emb2, batch_processing=False):
        """
        Perform bidaf and return the updated emb2.
        This method can handle single data points as well as batches.
        :param emb1: (batch, x_len, hidden_size)
        :param emb2: (batch, y_len, hidden_size)
        :return: (batch, y_len, output_size)
        """

        # make sure that batch processing works, even for single data points
        emb1 = emb1.unsqueeze(0) if len(emb1.shape) < 3 else emb1
        emb2 = emb2.unsqueeze(0) if len(emb2.shape) < 3 else emb2

        x_len = emb1.size(1) # (batch, x_len, hidden_size)
        y_len = emb2.size(1) # (batch, y_len, hidden_size)

        xy = []
        for i in range(x_len):
            xi = emb1.select(1, i).unsqueeze(1)  # (batch, 1, hidden_size)
            yi = self.att_weight_cq(emb2 * xi).squeeze(-1) # (batch, y_len, 1) --> (batch, y_len)
            xy.append(yi) # (x_len, batch, y_len)
        xy = torch.stack(xy, dim=-1)  # (batch, y_len, x_len)

        # (batch, y_len, x_len)
        s = self.att_weight_c(emb2).expand(-1, -1, x_len) + \
            self.att_weight_q(emb1).permute(0, 2, 1).expand(-1, y_len, -1) + \
            xy

        a = nnF.softmax(s, dim=2)  # (batch, y_len, x_len)

        # (batch, y_len, x_len) * (batch, x_len, hidden_size) -> (batch, y_len, hidden_size)
        y2x_att = torch.bmm(a, emb1)

        b = nnF.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1) # (batch, 1, y_len)

        # (batch, 1, y_len) * (batch, y_len, hidden_size) -> (batch, hidden_size)
        x2y_att = torch.bmm(b, emb2).squeeze(1)

        # (batch, y_len, hidden_size) (tiled)
        x2y_att = x2y_att.unsqueeze(1).expand(-1, y_len, -1)

        # (batch, y_len, hidden_size * 4)
        z = torch.cat([emb2, y2x_att, emb2 * y2x_att, emb2 * x2y_att], dim=-1)
        z = self.reduction_layer(z)  # (batch, y_len, output_size)

        return z.squeeze(0) if not batch_processing else z # (y_len, output_size) if no batch_processing








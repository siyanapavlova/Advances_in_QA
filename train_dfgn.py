"""
This is supposed to do the whole job!
(combine all modules, train some of them, maybe evaluate)
"""

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
import utils


class DFGN(torch.nn.Module): # TODO extract this to a separate module
    #TODO? implement loading of a previously trained DFGN model (for final evaluation!) ?
    def __init__(self, text_length, emb_size, device=torch.device('cpu'),
                 fb_dropout=0.5, predictor_dropout=0.3):
        #TODO docstring
        super(DFGN, self).__init__() #TODO pass the device to the Encoder and the Predictor as well?
        self.encoder = Encoder.Encoder(text_length=text_length)
        self.fusionblock = FusionBlock.FusionBlock(emb_size, device=device, dropout=fb_dropout) #TODO sort out init
        self.predictor = Predictor.Predictor(text_length, emb_size, dropout=predictor_dropout) #TODO sort out init

    def from_file(self, filepath):
        #TODO load all 3 parts individually and then make a DFGN object? Or des save()/load() handle this well already?
        pass

    def forward(self, query_ids, context_ids, graph, fb_passes):
        """
        #TODO docstring
        :param query_ids: Tensor[int] -- token IDs from Encoder.tokenizer
        :param context_ids: Tensor[int] -- token IDs from Encoder.tokenizer
        :param graph:
        :param fb_passes:
        :return:
        """
        #print(f"in train.dfgn: device of query_ids/context_ids: {str(query_ids.device)}, {ste(context_ids.device)}") #CLEANUP

        # forward through encoder
        q_emb = self.encoder(context_ids, query_ids)
        c_emb = self.encoder(query_ids, context_ids)

        # forward through fb
        Ct = self.fusionblock(c_emb, q_emb, graph, passes=fb_passes)

        # forward through predictor: sup, start, end, type
        outputs = self.predictor(Ct) # (1,M,1), (1,M,1), (1,M,1), (1, 3)

        return outputs

def train(net, train_data, #dev_data,
          dev_data_filepath, dev_preds_filepath, model_save_path,
          para_selector, #TODO sort these nicely
          ps_threshold=0.1,
          ner_device=torch.device('cpu'), training_device=torch.device('cpu'),
          text_length=250,
          fb_passes=1, coefs=(0.5, 0.5),
          epochs=3, batch_size=1, learning_rate=1e-4,
          eval_interval=None, verbose_evaluation=False, timed=False):
    #TODO docstring
    """

    :param net: DFGN object
    :param train_data: training data (raw points), split into batches
    :param dev_data: data for evaluation during training (raw points), split into batches
    :param model_save_path:
    :param text_length: limit the context's number of tokens (used in ParagraphSelector and EntityGraph)
    :param fb_passes: number of passes through the fusion block (fb)
    :param coefs: (float,float) coefficients for optimization (formula 15)
    :param epochs:
    :param batch_size:
    :param learning_rate:
    :param eval_interval:
    :return: list[(real_batch_size, overall_loss, sup_loss, start_loss, end_loss, type_loss)], list[dict{metrics}], Timer
    """
    timer = utils.Timer()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    flair.device = torch.device(ner_device)
    ner_tagger = flair.models.SequenceTagger.load('ner') # this hard-codes flair tagging!

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    losses = []
    real_batch_sizes = []  # some data points are not usable; this logs the real sizes
    dev_scores = []

    # Set the network into train mode
    net.train()
    net = net.to(training_device)

    timer("training_preparation")

    print("Training...")


    best_score = 0
    eval_interval = eval_interval if eval_interval else float('inf') # interval in batches
    a_model_was_saved_at_some_point = False

    for epoch in range(epochs):
        # TODO take recurrent times for forward, evaluation saving etc.
        print('Epoch %d/%d' % (epoch + 1, epochs))
        batch_counter = 0

        for step, batch in enumerate(tqdm(train_data, desc="Iteration")):

            """ DATA PROCESSING """
            ids = []
            queries = []
            contexts = []
            graphs = []

            useless_datapoint_inds = []

            for i, point in enumerate(batch):

                # make a list[ list[str, list[str]] ] for each point in the batch
                context = para_selector.make_context(point,
                                                       threshold=ps_threshold,
                                                       context_length=text_length) #TODO add device and numerated arguments
                graph = EntityGraph.EntityGraph(context,
                                                  context_length=text_length,
                                                  tagger=ner_tagger)
                if graph.graph:
                    ids.append(point[0])
                    queries.append(point[2])
                    contexts.append(context)
                    graphs.append(graph)
                else: # if the NER in EntityGraph doesn't find entities, the datapoint is useless.
                    useless_datapoint_inds.append(i)

            batch = [point for point in batch if point[0] in ids] # update the batch to exclude useless data points

            real_batch_sizes.append(batch_size - len(useless_datapoint_inds)) #TODO track the batch sizes!

            # if our batch is completely useless, just continue with the next batch. :(
            if len(useless_datapoint_inds) == batch_size:
                continue

            # turn the texts into tensors in order to put them on the GPU
            qc_ids = [net.encoder.token_ids(q, c) for q, c in zip(queries, contexts)] # list[ (list[int], list[int]) ]
            q_ids, c_ids = list(zip(*qc_ids)) # tuple(list[int]), tuple(list[int])
            q_ids_list = [torch.tensor(q) for q in q_ids] # list[Tensor] #TODO? maybe put this into forward()?
            c_ids_list = [torch.tensor(c) for c in c_ids] # list[Tensor]

            """ MAKE TRAINING LABELS """
            # replace the paragraphs in raw_point with their shortened versions (obtained from PS)
            for (i, p), c in zip(enumerate(batch), contexts):
                batch[i][3] = c

            # TODO? change utils.make_labeled_data_for_predictor() to process batches of data?
            labels = [utils.make_labeled_data_for_predictor(g,p,tokenizer) for g,p in zip(graphs, batch)] # list[(support, start, end, type)]
            # list[(Tensor, Tensor, Tensor, Tensor)] -> tuple(Tensor), tuple(Tensor), tuple(Tensor), tuple(Tensor)
            sup_labels, start_labels, end_labels, type_labels = list(zip(*labels))
            #print(f"in train_dfgn.train(): shapes of labels:\n{len(sup_labels)}, {len(start_labels)}, {len(end_labels)}, {len(type_labels)}") #CLEANUP

            q_ids_list = [t.to(training_device) if t is not None else None for t in q_ids_list]
            c_ids_list = [t.to(training_device) if t is not None else None for t in c_ids_list]
            for i, g in enumerate(graphs):
                graphs[i].M = g.M.to(training_device) # work with enumerate to actually mutate the graph objects

            sup_labels = torch.stack(sup_labels).to(training_device)      # (batch, M)
            start_labels = torch.stack(start_labels).to(training_device)  # (batch, 1)
            end_labels = torch.stack(end_labels).to(training_device)      # (batch, 1)
            type_labels = torch.stack(type_labels).to(training_device)    # (batch)


            """ FORWARD PASSES """
            optimizer.zero_grad()


            sups, starts, ends, types = [], [], [], []
            for query, context, graph in zip(q_ids_list, c_ids_list, graphs): # 'graph' is not a tensor -> for-loop instead of batch processing

                o_sup, o_start, o_end, o_type = net(query, context, graph, fb_passes=fb_passes) # (M, 2), (M), (M), (1, 3)
                sups.append(o_sup)
                starts.append(o_start)
                ends.append(o_end)
                types.append(o_type)

            sups =   torch.stack(sups)   # (batch, M, 2)
            starts = torch.stack(starts) # (batch, 1, M)
            ends =   torch.stack(ends)   # (batch, 1, M)
            types =  torch.stack(types)  # (batch, 1, 3)

            """ LOSSES & BACKPROP """
            weights = torch.ones(2, device=training_device) #TODO maybe extract this to a tiny function?
            sup_label_batch = sup_labels.view(-1)
            weights[0] = sum(sup_label_batch)/float(sup_label_batch.shape[0])
            weights[1] -= weights[0] # assign the opposite weight

            sup_criterion = torch.nn.CrossEntropyLoss(weight=weights)
            criterion = torch.nn.CrossEntropyLoss()  # for prediction of answer type

            # use .view(-1,...) to put points together (this is like summing the points' losses)
            sup_loss =   sup_criterion(sups.view(-1,2), sup_label_batch) # (batch*M, 2), (batch*M)
            start_loss = sum([criterion(starts[i], start_labels[i]) for i in range(start_labels.shape[0])])  # batch * ( (1, M, 1), (1) )
            end_loss   = sum([criterion(ends[i], end_labels[i]) for i in range(end_labels.shape[0])])        # batch * ( (1, M, 1), (1) )
            type_loss  =  criterion(types.view(-1,3),  type_labels.view(-1))    # (batch, 1, 3), (batch, 1)

            # This doesn't have the weak supervision BFS mask stuff from section 3.5 of the paper
            #TODO? maybe start training with start/end loss only first, then train another model on all 4 losses?
            loss = start_loss + end_loss + coefs[0]*sup_loss + coefs[1]*type_loss # formula 15

            loss.backward(retain_graph=True)
            losses.append( (loss.item(),
                            sup_loss.item(),
                            start_loss.item(),
                            end_loss.item(),
                            type_loss.item()) ) # for logging purposes

            batch_counter += 1
            # Evaluate on validation set after some iterations
            if batch_counter % eval_interval == 0:

                # this calls the official evaluation script (altered to return metrics)
                metrics = evaluate(net, #TODO make this prettier
                                   tokenizer, ner_tagger,
                                   training_device, dev_data_filepath, dev_preds_filepath,
                                   fb_passes = fb_passes,
                                   text_length = text_length,
                                   verbose=verbose_evaluation)
                score = metrics["joint_f1"]
                dev_scores.append(metrics) # appends the whole dict of metrics
                if score > best_score:
                    print(f"Better eval found with accuracy {round(score, 3)} (+{round(score - best_score, 3)})")
                    best_score = score

                    torch.save(net, model_save_path) #TODO make sure that this works (maybe, should we save each of the 3 parts indvidually?)
                    a_model_was_saved_at_some_point = True
                else:
                    print(f"No improvement yet...")
                timer(f"training_evaluation_{batch_counter/eval_interval}")



            optimizer.step()
        timer(f"training_epoch_{epoch}")

    if not a_model_was_saved_at_some_point:  # make sure that there is a model file
        print(f"saving model to {model_save_path}...")
        torch.save(net, model_save_path) #TODO make sure that this works (maybe have a method that saves all 3 parts individually?)

    losses_with_batchsizes = [(b, t[0], t[1], t[2], t[3], t[4]) for b,t in zip(real_batch_sizes, losses)]
    return (losses_with_batchsizes, dev_scores, timer) if timed else (losses_with_batchsizes, dev_scores)

def predict(net, query, context, graph, tokenizer, sentence_lengths, fb_passes=1):
    """
    # TODO docstring
    #TODO make this a method of DFGN
    :param net:
    :param query:
    :param context:
    :param graph:
    :param tokenizer:
    :param sentence_lengths:
    :param fb_passes:
    :return:
    """

    # (M,2), (1,M), (1,M), (1,3)
    o_sup, o_start, o_end, o_type = net(query, context, graph, fb_passes=fb_passes)

    # =========== GET ANSWERS
    answer_start = o_start.argmax() #TODO make sure that these tensors are all only containing one number!
    answer_end = o_end.argmax()
    answer_type = o_type.argmax()
    if answer_type == 0:
        answer = "yes"
    elif answer_type == 1:
        answer = "no"
    elif answer_type == 2 and answer_end >= answer_start:
        answer = tokenizer.decode(graph.tokens[answer_start: answer_end + 1])
    else:
        answer = "noanswer"

    # =========== GET SUPPORTING FACTS
    pos = 0
    sup_fact_pairs = []
    for para, s_lens in zip(context, sentence_lengths):
        for j, s_len in enumerate(s_lens):
            #score = round(sum(o_sup.argmax([pos: pos + s_len])) / s_len)
            # take avg of token-wise scores and round to 0 or 1

            score = round(float(sum([x.argmax() for x in o_sup.T[pos: pos + s_len]]) / float(s_len)))
            if score == 1:
                sup_fact_pairs.append([para[0], j])
            pos += s_len

    return answer, sup_fact_pairs


def evaluate(net,
             tokenizer, ner_tagger,
             device, eval_data_filepath, eval_preds_filepath,
             fb_passes = 1, text_length = 250, verbose=False):
    """
    #TODO docstring
    :param net:
    :param dev_data:
    :return:
    """

    """ PREPADE DATA FOR PREDICTION """
    dh = utils.HotPotDataHandler(eval_data_filepath)
    dev_data = dh.data_for_paragraph_selector()

    point_ids = [point[0] for point in dev_data] # needed to handle useless datapoints
    queries = [point[2] for point in dev_data]
    contexts = [point[3] for point in dev_data]

    graphs = [EntityGraph.EntityGraph(c,
                                      context_length=text_length,
                                      tagger=ner_tagger)
              for c in contexts]

    # if the NER in EntityGraph doesn't find entities, the datapoint is useless.
    useless_datapoint_inds = [i for i, g in enumerate(graphs) if not g.graph]
    queries = [q for i, q in enumerate(queries) if i not in useless_datapoint_inds]
    contexts = [c for i, c in enumerate(contexts) if i not in useless_datapoint_inds]
    graphs = [g for i, g in enumerate(graphs) if i not in useless_datapoint_inds]


    # required for prediction in the right format
    s_lens_batch = [utils.sentence_lengths(c, tokenizer) for c in contexts]

    # turn the texts into tensors in order to put them on the GPU
    qc_ids = [net.encoder.token_ids(q, c) for q, c in zip(queries, contexts)]  # list[ (list[int], list[int]) ]
    q_ids, c_ids = list(zip(*qc_ids))  # tuple(list[int]), tuple(list[int])
    q_ids_list = [torch.tensor(q).to(device) for q in q_ids]  # list[Tensor]
    c_ids_list = [torch.tensor(c).to(device) for c in c_ids]  # list[Tensor]

    for i,g in enumerate(graphs):
        graphs[i].M = g.M.to(device)  # work with enumerate to actually mutate the graph objects


    """ FORWARD PASSES """
    answers = {}  # {question_id: str} (either "yes", "no" or a string containing the answer)
    sp = {}  # {question_id: list[list[paragraph_title, sent_num]]} (supporting sentences)

    # return useless datapoints unanswered
    for i in useless_datapoint_inds:
        answers[point_ids[i]] = "noanswer"
        sp[point_ids[i]] = []

    for i, (query, context, graph, s_lens) in enumerate(zip(q_ids_list, c_ids_list, graphs, s_lens_batch)):

        if verbose: print(queries[i])

        answer, sup_fact_pairs = predict(net, query, context, graph, tokenizer,
                                         s_lens, fb_passes=fb_passes) #TODO sort these parameters

        answers[dev_data[i][0]] = answer  # {question_id: str}
        sp[dev_data[i][0]] = sup_fact_pairs # {question_id: list[list[paragraph_title, sent_num]]}

        if verbose: print(answer)

    with open(eval_preds_filepath, 'w') as f:
        json.dump( {"answer":answers, "sp":sp} , f)


    """ EVALUATION """
    return official_eval_script.eval(eval_preds_filepath, eval_data_filepath) #TODO return aything else than the metrics?




if __name__ == '__main__':

    #=========== PARAMETER INPUT
    take_time = utils.Timer()

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', metavar='config', type=str,
                        help='configuration file for training')
    parser.add_argument('model_name', metavar='model', type=str,
                        help="name of the model's file")
    args = parser.parse_args()

    #TODO make sure that the config contains all required parameters
    cfg = utils.ConfigReader(args.config_file)

    #TODO change path assignment to fit with the program
    model_abs_path = cfg('model_abs_dir') + args.model_name + "/"
    model_filepath = model_abs_path + args.model_name
    losses_abs_path = model_abs_path + "losses" # contains (batch_size, overall_loss, sup_l., start_l., end_l., type_l.)
    traintime_abs_path = model_abs_path + "times"
    devscores_abs_path = model_abs_path + "devscores"

    eval_data_dump_dir = cfg("eval_data_dump_dir")
    eval_data_dump_filepath =  eval_data_dump_dir + "gold"
    eval_preds_dump_filepath = eval_data_dump_dir + "predictions"

    # check all relevant file paths and directories before starting training
    # make sure that the training data will be found
    for path in [cfg("data_abs_path"), cfg("dev_data_abs_path")]:
        try:
            f = open(path, "r")
            f.close()
        except FileNotFoundError as e:
            print(e)
            sys.exit()
    # make sure that the output directories exist
    for path in [model_abs_path, eval_data_dump_dir, cfg("ps_model_abs_path")]:
        if not os.path.exists(path):
            print(f"newly creating {path}")
            os.makedirs(path)

    take_time("parameter input")


    #========== DATA PREPARATION
    # For training, load data from the HotPotQA training set (or a subset that was
    # previously pickled) and split off some data for evaluation during training.
    # The HotPotQA dev set is reserved for evaluation and thus not used here.

    # try to load pickled data, and in case it doesn't work, read the whole HotPotQA dataset
    try:
        with open(cfg("pickled_train_data"), "rb") as f:
            train_data_raw = pickle.load(f)
            training_dataset_size = cfg("training_dataset_size") if cfg("training_dataset_size") else len(train_data_raw)
        with open(cfg("pickled_dev_data"), "rb") as f:
            dev_data_raw = pickle.load(f)
            # restrict loaded dev data to the required percentage
            dev_data_raw = dev_data_raw[:cfg('percent_for_eval_during_training') * training_dataset_size]

    except: #TODO why does it always go to this exception instead of loading the pickled data?
        print(f"Reading data from {cfg('data_abs_path')}...") # the whole HotPotQA training set
        dh = utils.HotPotDataHandler(cfg("data_abs_path"))
        raw_data = dh.data_for_paragraph_selector() # get raw points
        training_dataset_size = cfg("training_dataset_size") if cfg("training_dataset_size") else len(raw_data)

        print("Splitting data...") # split into what we need DURING training
        train_data_raw, dev_data_raw = train_test_split(raw_data[:training_dataset_size],  # restricts the number of training+dev examples
                                                        test_size=cfg('percent_for_eval_during_training'),
                                                        random_state=cfg('shuffle_seed'),
                                                        shuffle=True)

        #print(f"in train_dfgn.main(): len(dev_data_raw): {len(dev_data_raw)}") #CLEANUP
        #print(dev_data_raw)
        #train_data = ParagraphSelector.make_training_data(train_data_raw, text_length=cfg("text_length")) #CLEANUP
        #train_data = shuffle(train_data, random_state=cfg('data_shuffle_seed')) #CLEANUP?

        with open(cfg("pickled_train_data"), "wb") as f:
            pickle.dump(train_data_raw, f)

        with open(cfg("pickled_dev_data"), "wb") as f:
            pickle.dump(dev_data_raw, f)



    # group training data into batches
    bs = cfg("batch_size")
    N = len(train_data_raw)
    train_data_raw = [train_data_raw[i : i+bs] for i in range(0, N, bs)]

    take_time("data preparation")


    training_device = torch.device('cpu')
    if cfg("try_training_on_gpu") and torch.cuda.is_available():
        torch.cuda.set_device(cfg("gpu_number"))
        training_device = torch.device('cuda')
    tagger_device = torch.device('cuda') if cfg("use_gpu_for_ner") else torch.device('cpu')


    # ========== DFGN START

    para_selector = ParagraphSelector.ParagraphSelector(cfg("ps_model_abs_path"))

    dh.make_eval_data(para_selector,
                      dev_data_raw,
                      eval_data_dump_filepath,
                      cfg)
    take_time("eval_data preparation")

    dfgn = DFGN(text_length=cfg("text_length"),
                emb_size=cfg("emb_size"),
                device=training_device,
                fb_dropout=cfg("fb_dropout"),
                predictor_dropout=cfg("predictor_dropout"))

    losses, dev_scores, train_times = train(dfgn, #TODO watch out with the parameter sorting!
                          train_data_raw, # in batches
                          #dev_data_raw, # not needed for pre-processing during evaluation
                          eval_data_dump_filepath, # for reading processed dev_data_raw
                          eval_preds_dump_filepath, # for dumping predictions during evaluation
                          model_filepath, # where the dfgn model will be saved
                          para_selector,
                          ps_threshold=cfg("ps_threshold"),
                          ner_device=tagger_device,
                          training_device=training_device,
                          fb_passes=cfg("fb_passes"),
                          coefs=(cfg("lambda_s"), cfg("lambda_t")),
                          text_length=cfg("text_length"),
                          epochs=cfg("epochs"),
                          batch_size=cfg("batch_size"),
                          learning_rate=cfg("learning_rate"),
                          eval_interval=cfg("eval_interval"),
                          verbose_evaluation=cfg("verbose_evaluation"),
                          timed=True)

    take_time("training")


    # ========== LOGGING
    print(f"Saving losses in {losses_abs_path}...")
    with open(losses_abs_path, "w") as f:
        f.write("batch_size\toverall_loss\tsup_loss\tstart_loss\tend_loss\ttype_loss\n")
        f.write("\n".join(["\t".join([str(l) for l in step]) for step in losses]))

    print(f"Saving dev scores in {devscores_abs_path}...")
    with open(devscores_abs_path, "w") as f:
        for metrics_dict in dev_scores:
            f.write(str(metrics_dict)) # this can be parsed with ast.literal_eval() later on

    print(f"Saving config and times taken to {traintime_abs_path}...")
    with open(traintime_abs_path, 'w', encoding='utf-8') as f:
        f.write("Configuration in: " + args.config_file + "\n")
        f.write(str(cfg)+"\n")

        f.write("\n Times taken per step:\n" + str(train_times) + "\n")
        take_time("saving results")
        take_time.total()
        f.write("\n Overall times taken:\n" + str(take_time) + "\n")

    print("\nTimes taken:\n", take_time)
    print("done.")



























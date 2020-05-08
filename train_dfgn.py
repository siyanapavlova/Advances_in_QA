"""
This is supposed to do the whole job!
(combine all modules, train some of them, maybe evaluate)
"""

import os, sys, argparse
import pickle # mainly for training data
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

import flair # for NER in the EntityGraph
from modules import ParagraphSelector, EntityGraph, Encoder, FusionBlock, Predictor
import utils

#TODO 2020-05-07: implement periodic evaluation during training; test dfgn; start big training

class DFGN(torch.nn.Module):
    #TODO implement a save() method?
    #TODO implement loading of a previously trained DFGN model
    #TODO implement a predict() method for testing
    def __init__(self, text_length, emb_size,
                 fb_dropout=0.5, predictor_dropout=0.3):
        #TODO docstring
        super(DFGN, self).__init__()
        self.encoder = Encoder.Encoder(text_length=text_length)
        self.fusionblock = FusionBlock.FusionBlock(emb_size, dropout=fb_dropout) #TODO sort out init
        self.predictor = Predictor.Predictor(text_length, emb_size, dropout=predictor_dropout) #TODO sort out init

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

def train(net, train_data, dev_data, model_save_path,
          ps_path, ps_threshold=0.1,
          ner_with_gpu=False, try_training_on_gpu=True,
          text_length=250,
          fb_passes=1, coefs=(0.5, 0.5),
          epochs=3, batch_size=1, learning_rate=1e-4,
          eval_interval=None, timed=False):
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
    :return:
    """
    timer = utils.Timer()

    para_selector = ParagraphSelector.ParagraphSelector(ps_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #TODO initialize this once only (= extract it to the main method)

    tagger_device = 'gpu' if ner_with_gpu else 'cpu'
    flair.device = torch.device(tagger_device)
    ner_tagger = flair.models.SequenceTagger.load('ner') # this hard-codes flair tagging!

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    losses = []

    # Set the network into train mode
    net.train()

    cuda_is_available = torch.cuda.is_available() if try_training_on_gpu else False
    device = torch.device('cuda') if cuda_is_available else torch.device('cpu')
    # put the net on the GPU if possible
    if cuda_is_available:
        net = net.to(device)

    timer("training_preparation")

    print("Training...")

    #c = 0  # counter over training examples #CLEANUP?
    #best_acc = 0
    #eval_interval = eval_interval if eval_interval else float('inf')
    #batched_interval = round(eval_interval / batch_size)  # number of batches needed to reach eval_interval
    a_model_was_saved_at_some_point = False

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        lost_datapoints_log = []

        for step, batch in enumerate(tqdm(train_data, desc="Iteration")):

            """ DATA PROCESSING """
            queries = [point[2] for point in batch]

            # make a list[ list[str, list[str]] ] for each point in the batch
            contexts = [para_selector.make_context(point,
                                                   threshold=ps_threshold,
                                                   context_length=text_length)
                        for point in batch]

            graphs = [EntityGraph.EntityGraph(c,
                                              context_length=text_length,
                                              tagger=ner_tagger)
                      for c in contexts]

            # if the NER in EntityGraph doesn't find entities, the datapoint is useless.
            useless_datapoint_inds = [i for i,g in enumerate(graphs) if not g.graph]
            queries = [q for i,q in enumerate(queries) if i not in useless_datapoint_inds]
            contexts = [c for i, c in enumerate(contexts) if i not in useless_datapoint_inds]
            graphs = [g for i, g in enumerate(graphs) if i not in useless_datapoint_inds]
            lost_datapoints_log.append(len(useless_datapoint_inds)) #TODO track the batch sizes!
            print(f"number of useless datapoints in this batch: {len(useless_datapoint_inds)}/{batch_size}") #CLEANUP

            # if our batch is completely useless, just continue with the next batch. :(
            if len(useless_datapoint_inds) == batch_size:
                continue

            # turn the texts into tensors in order to put them on the GPU
            qc_ids = [net.encoder.token_ids(q, c) for q, c in zip(queries, contexts)] # list[ (list[int], list[int]) ]
            q_ids, c_ids = list(zip(*qc_ids)) # tuple(list[int]), tuple(list[int])
            q_ids_list = [torch.tensor(q) for q in q_ids] # list[Tensor]
            c_ids_list = [torch.tensor(c) for c in c_ids] # list[Tensor]

            """ MAKE TRAINING LABELS """
            # this is a list of 4-tuples: (support, start, end, type)
            #TODO change utils.make_labeled_data_for_predictor() to process batches of data
            labels = [utils.make_labeled_data_for_predictor(g,p,tokenizer) for g,p in zip(graphs, batch)]
            # list[(Tensor, Tensor, Tensor, Tensor)] -> tuple(Tensor), tuple(Tensor), tuple(Tensor), tuple(Tensor)
            sup_labels, start_labels, end_labels, type_labels = list(zip(*labels))

            q_ids_list = [t.to(device) if t is not None else None for t in q_ids_list]
            c_ids_list = [t.to(device) if t is not None else None for t in c_ids_list]
            for graph in graphs: graph.M = graph.M.to(device)
            sup_labels = [t.to(device) if t is not None else None for t in sup_labels] #TODO change this once it's done in the utils function
            start_labels = [t.to(device) if t is not None else None for t in start_labels]
            end_labels = [t.to(device) if t is not None else None for t in end_labels]
            type_labels = [t.to(device) if t is not None else None for t in type_labels]

            optimizer.zero_grad()

            """ FORWARD PASSES """
            # As 'graph' is not a tensor, normal batch processing isn't possible
            sups, starts, ends, types = [], [], [], []
            for query, context, graph in zip(q_ids_list, c_ids_list, graphs):
                # (M,2), (M,2), (M,2), (3)
                o_sup, o_start, o_end, o_type = net(query, context, graph, fb_passes=fb_passes)
                sups.append(o_sup)
                starts.append(o_start)
                ends.append(o_end)
                types.append(o_type)

            """ LOSSES & BACKPROP """
            sup_loss = sum([criterion(p,l) for p,l in zip(sups, sup_labels)])
            start_loss = sum([criterion(p,l) for p,l in zip(starts, start_labels)])
            end_loss = sum([criterion(p,l) for p,l in zip(ends, end_labels)])
            type_loss = sum([criterion(p,l) for p,l in zip(types, type_labels)])

            loss = start_loss + end_loss + coefs[0]*sup_loss + coefs[1]*type_loss # formula 15

            loss.backward(retain_graph=True)
            losses.append(loss.item()) # for logging purposes #TODO log all 4 individual losses separately?

            #c += 1 #TODO include this part
            ## Evaluate on validation set after some iterations
            #if c % batched_interval == 0:
            #    _, _, _, accuracy, _, _, _ = self.evaluate(dev_data)

            #   if accuracy > best_acc:
            #        print(
            #            f"Better eval found with accuracy {round(accuracy, 3)} (+{round(accuracy - best_acc, 3)})")
            #        best_acc = accuracy
            #        torch.save(net, model_save_path)
            #        a_model_was_saved_at_some_point = True
            #    else:
            #        print(f"No improvement yet...")
            #    timer(f"training_evaluation_{c/batched_interval}")

            optimizer.step()
        timer(f"training_epoch_{epoch}")

    if not a_model_was_saved_at_some_point:  # make sure that there is a model file
        torch.save(net, model_save_path) #TODO make sure that this works

    return (losses, timer) if timed else losses




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
    #model_abs_path += '.pt' if not args.model_name.endswith('.pt') else '' #CLEANUP
    losses_abs_path = model_abs_path + "losses"
    traintime_abs_path = model_abs_path + "times"

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
    for path in [model_abs_path]:
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




    # ========== DFGN START

    dfgn = DFGN(text_length=cfg("text_length"),
                emb_size=cfg("emb_size"),
                fb_dropout=cfg("fb_dropout"),
                predictor_dropout=cfg("predictor_dropout")) #TODO make sure this works; maybe include parameters from the config?

    #TODO extract the loading of the NER tagger to somewhere outside training/batches
    losses, times = train(dfgn,
                          train_data_raw, # in batches
                          dev_data_raw,
                          model_filepath, # where the dfgn model will be saved
                          ps_path=cfg("ps_model_abs_path"),
                          ps_threshold=cfg("ps_threshold"),
                          ner_with_gpu=cfg("use_gpu_for_ner"),
                          try_training_on_gpu=cfg("try_training_on_gpu"),
                          fb_passes=cfg("fb_passes"),
                          coefs=(cfg("lambda_s"), cfg("lambda_t")),
                          text_length=cfg("text_length"),
                          epochs=cfg("epochs"),
                          batch_size=cfg("batch_size"),
                          learning_rate=cfg("learning_rate"),
                          eval_interval=cfg("eval_interval"),
                          timed=True)

    take_time("training")



    take_time.total()
    print("times for training:")
    print(times)
    print("\noverall times:")
    print(take_time)
    print("done.")



























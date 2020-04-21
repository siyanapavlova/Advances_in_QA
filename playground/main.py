"""
This is supposed to do the whole job!
(combine all modules, train some of them, maybe evaluate)
"""

import os, sys, argparse
import pickle # mainly for training data
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split


from modules import ParagraphSelector, EntityGraph, Encoder, FusionBlock, Predictor
import utils


#TODO config input, blablabla


class DFGN(torch.nn.Module):
    def __init__(self):
        super(DFGN, self).__init__()
        self.encoder = Encoder(text_length)
        self.fusionblock = FusionBlock(emb_size) #TODO sort out init
        self.predictor = Predictor(text_length, emb_size) #TODO sort out init

    def forward(self, query, context, graph, text_length, fb_passes):
        #TODO docstring

        # forward through encoder
        q_ids, c_ids = self.encoder.token_ids(query, context)
        q_emb = self.encoder.predict(c_ids, q_ids)
        c_emb = self.encoder.predict(q_ids, c_ids)

        # forward through fb
        Ct = self.fusionblock(c_emb, q_emb, graph, passes=fb_passes)

        # forward through predictor: sup, start, end, type
        outputs = self.predictor(Ct) # (1,M,1), (1,M,1), (1,M,1), (1, 3)

        return outputs



def train(net, train_data, dev_data, model_save_path, ps_path,
          ps_threshold=0.1,
          text_length=250, fb_passes=1, coefs=(0.5, 0.5),
          epochs=10, batch_size=1, learning_rate=0.0001, eval_interval=None):
    #TODO docstring
    """

    :param net:
    :param train_data:
    :param dev_data:
    :param model_save_path:
    :param text_length:
    :param fb_passes:
    :param coefs: (float,float) coefficients for optimization (formula 15)
    :param epochs:
    :param batch_size:
    :param learning_rate:
    :param eval_interval:
    :return:
    """

    #======= START OF COPY =======#
    para_selector = ParagraphSelector(ps_path)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    losses = []

    # Set the network into train mode
    net.train()

    cuda_is_available = torch.cuda.is_available()
    device = torch.device('cuda') if cuda_is_available else torch.device('cpu')

    # put the net on the GPU if possible
    if cuda_is_available:
        net = net.to(device)

    print("Training...")

    #c = 0  # counter over training examples #CLEANUP?
    #best_acc = 0
    #eval_interval = eval_interval if eval_interval else float('inf')
    #batched_interval = round(eval_interval / batch_size)  # number of batches needed to reach eval_interval
    a_model_was_saved_at_some_point = False

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))

        for step, batch in enumerate(tqdm(train_data, desc="Iteration")):

            batch = [t.to(device) if t is not None else None for t in batch]

            queries = [point[2] for point in batch]
            contexts = [para_selector.make_context(point,
                                                   threshold=ps_threshold) for point in batch]
            graphs = [EntityGraph(c) for c in contexts]
            # this is a list of 4-tuples: (support, start, end, type)
            # of shape (M, M, M, 3) with M = number of tokens in raw point
            labels = [utils.make_labeled_data_for_predictor(g,p) for g,p in zip(graphs, batch)]

            #TODO is it necessary to put everything onto the GPU?
            queries = [t.to(device) if t is not None else None for t in queries]
            contexts = [t.to(device) if t is not None else None for t in contexts]
            graphs = [t.to(device) if t is not None else None for t in graphs]
            labels = [t.to(device) if t is not None else None for t in labels]

            optimizer.zero_grad()

            # As 'graph' is not a tensor, normal batch processing isn't possible
            #TODO 21.4.2020: continue here.
            # - sort out parameter passes
            # - sort out batch processing
            # - clean up the old functions and copied parts
            # - re-activate periodical evaluation?
            # - ......
            sups, starts, ends, types = [], [], [], []
            for query, context, graph in zip(queries, contexts, graphs):
                # (1,M,1), (1,M,1), (1,M,1), (1, 3)
                o_sup, o_start, o_end, o_type = net(query, context, graph,
                                                    text_length=text_length,
                                                    fb_passes=fb_passes).squeeze(1)  # TODO keep squeeze(1)? If so, why?
                sups.append(o_sup)
                starts.append(o_start)
                ends.append(o_end)
                types.append(o_type)

            sup_loss = criterion(sups, labels[0])
            start_loss = criterion(starts, labels[1])
            end_loss = criterion(ends, labels[2])
            type_loss = criterion(types, labels[3])

            loss = start_loss + end_loss + coefs[0]*sup_loss + coefs[1]*type_loss # formula 15

            loss.backward(retain_graph=True)
            losses.append(loss.item())

            #c += 1 #TODO include this part
            ## Evaluate on validation set after some iterations
            #if c % batched_interval == 0:
            #    _, _, _, accuracy, _, _, _ = self.evaluate(dev_data)

            #   if accuracy > best_acc:
            #        print(
            #            f"Better eval found with accuracy {round(accuracy, 3)} (+{round(accuracy - best_acc, 3)})")
            #        best_acc = accuracy
            #        self.net.save_pretrained(model_save_path)
            #        a_model_was_saved_at_some_point = True
            #    else:
            #        print(f"No improvement yet...")

            optimizer.step()

    if not a_model_was_saved_at_some_point:  # make sure that there is a model file
        net.save(model_save_path) #TODO make sure that this works

    return losses
    #======= END OF COPY =======#





#TODO delete the below
def PARAGRAPHSELECTORtrain(self, train_data, dev_data, model_save_path,
          epochs=10, batch_size=1, learning_rate=0.0001, eval_interval=None):
    """
    Train a ParagraphSelectorNet on a training dataset.
    Binary Cross Entopy is used as the loss function.
    Adam is used as the optimizer.

    :param train_data: a tensor as returned by the make_training_data() function;
                       it has two columns:
                    a train tensor with two columns:
                        1. token_ids as returned by the tokenizer for
                           [CLS] + query + [SEP] + paragraph + [SEP]
                           (10 entries per datapoint, one of each paragraph)
                        2. labels for the points - 0 if the paragraphs is
                           no relevant to the query, and 1 otherwise
    :param epochs: number of training epochs, default is 10
    :param batch_size: batch size for the training, default is 1
    :param learning_rate: learning rate for the optimizer,
                          default is 0.0001

    :return losses: a list of losses
    """
    # Use Binary Cross Entropy as a loss function instead of MSE
    # There are papers on why MSE is bad for classification
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    losses = []

    # Set the network into train mode
    self.net.train()

    cuda_is_available = torch.cuda.is_available()
    device = torch.device('cuda') if cuda_is_available else torch.device('cpu')

    # put the net on the GPU if possible
    if cuda_is_available:
        self.net = self.net.to(device)

    print("Training...")

    # TODO: find a way to shuffle reproducibly
    train_data = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    c = 0  # counter over taining examples
    best_acc = 0
    eval_interval = eval_interval if eval_interval else float('inf')
    batched_interval = round(eval_interval / batch_size)  # number of batches needed to reach eval_interval
    a_model_was_saved_at_some_point = False

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))

        for step, batch in enumerate(tqdm(train_data, desc="Iteration")):
            batch = [t.to(device) if t is not None else None for t in batch]
            inputs, labels = batch

            optimizer.zero_grad()

            outputs = self.net(inputs).squeeze(1)  # TODO why squeeze(1)?
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            losses.append(loss.item())

            c += 1
            # Evaluate on validation set after some iterations
            if c % batched_interval == 0:
                _, _, _, accuracy, _, _, _ = self.evaluate(dev_data)

                if accuracy > best_acc:
                    print(
                        f"Better eval found with accuracy {round(accuracy, 3)} (+{round(accuracy - best_acc, 3)})")
                    best_acc = accuracy
                    self.net.save_pretrained(model_save_path)
                    a_model_was_saved_at_some_point = True
                else:
                    print(f"No improvement yet...")

            optimizer.step()

    if not a_model_was_saved_at_some_point:  # make sure that there is a model file
        self.net.save_pretrained(model_save_path)

    return losses

















if __name__ == '__main__':

    #TODO take times throughout training
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
    #model_abs_path += '.pt' if not args.model_name.endswith('.pt') else ''
    losses_abs_path = model_abs_path + args.model_name + ".losses"
    traintime_abs_path = model_abs_path + args.model_name + ".times"

    # check all relevant file paths and directories before starting training
    # TODO change this
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


    #========== DATA PREPARATION
    #TODO update file paths here as well
    #TODO process the hotpotqa devset as well and pass it to the train() function
    try:
        with open(cfg("pickled_train_data"), "rb") as f:
            train_data = pickle.load(f)
            data_limit = cfg("dataset_size") if cfg("dataset_size") else len(train_data)
        with open(cfg("pickled_dev_data"), "rb") as f:
            dev_data = pickle.load(f)
            dev_data_limit = cfg("dev_data_limit") if cfg("dev_data_limit") else len(dev_data)

    except:
        print(f"Reading data from {cfg('data_abs_path')}...")
        dh = utils.HotPotDataHandler(cfg("data_abs_path"))
        data = dh.data_for_paragraph_selector()
        data_limit = cfg("dataset_size") if cfg("dataset_size") else len(data)

        dev_dh = utils.HotPotDataHandler(cfg("dev_data_abs_path"))
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



    # ========== DFGN START

    dfgn = DFGN() #TODO make sure this works; maybe include parameters from the config?

    losses = train(dfgn,
                   train_data,
                   dev_data[:dev_data_limit],
                   model_abs_path,
                   ps_path=cfg("ps_model_abs_path"),
                   fb_passes=cfg("fb_passes"),
                   coefs=(cfg("lambda_s"), cfg("lambda_t")),
                   text_length=cfg("text_length"),
                   epochs=cfg("epochs"),
                   batch_size=cfg("batch_size"),
                   learning_rate=cfg("learning_rate"),
                   eval_interval=cfg("eval_interval"))


























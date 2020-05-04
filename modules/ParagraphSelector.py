"""
This module implements the Paragraph Selector from the paper, Section 3.1
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from sklearn.utils import shuffle
import os,sys,inspect
import math
from tqdm import tqdm
import argparse

from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import HotPotDataHandler
from utils import ConfigReader
from utils import Timer

# weights for training, because we have imbalanced data:
# 80% of paragraphs are not important (= class 0) and 20% are important (class 1)
WEIGHTS = [0.2, 0.8] #CLEANUP? We implemented downscaling instead of this loss weighting


def make_training_data(data,
                       text_length=512,
                       tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
    """
    #TODO talk about downsampling!!! (only taking 2 of the uninformative paragraphs as negative data!)
    Make a train tensor for each datapoint.

    :param data: question ID, supporting facts, question, and paragraphs, 
                 as returned by HotPotDataHandler
    :type data: list(tuple(str, list(str), str, list(list(str, list(str)))))
    :param text_length: id of the pad token used to pad when
                        the paragraph is shorter then text_length
                        default is 0
    :param tokenizer: default: BertTokenizer(bert-base-uncased)

    :return: a train tensor with two columns:
                1. token_ids as returned by the tokenizer for
                   [CLS] + query + [SEP] + paragraph + [SEP]
                   (10 entries per datapoint, one of each paragraph)
                2. labels for the points - 0 if the paragraphs is
                   no relevant to the query, and 1 otherwise
    """

    neg_max = 2 # maximum number of useless paragraphs to be used per question
    labels = []
    datapoints = []
    for point in tqdm(data):
        neg_counter = 0
        for para in point[3]:
            is_useful_para = para[0] in point[1] # Label is 1: if paragraph title is in supporting facts, otherwise 0
            if not is_useful_para and neg_counter == neg_max: # enough negative examples
                continue
            else: # useful paragraph or neg_max not yet reached
                if not is_useful_para:
                    neg_counter += 1

                labels.append(float(is_useful_para))
                point_string = point[2] + " [SEP] " + ("").join(para[1])

                # automatically prefixes [CLS] and appends [SEP]
                token_ids = tokenizer.encode(point_string, max_length=512)

                # Add padding if there are fewer than text_length tokens,
                # else trim to text_length
                if len(token_ids) < text_length:
                    token_ids += [tokenizer.pad_token_id for _ in range(text_length - len(token_ids))]
                else:
                    token_ids = token_ids[:text_length]
                datapoints.append(token_ids)
        #print(sum(labels[-4:])==2) #CLEANUP

    # Turn labels and datapoints into tensors and put them together        
    label_tensor = torch.tensor(labels)
    train = torch.tensor(datapoints)
    train_tensor = torch.utils.data.TensorDataset(train, label_tensor)

    return train_tensor

class ParagraphSelector():
    """
    This class implements all that is necessary for training
    a paragraph selector model (as per the requirements in the 
    paper), predicting relevance scores for paragraph-query pairs
    and building the context for HotPotQA datapoint. Additionally,
    it also allows for saving a trained model, loading a trained 
    model from a file, and evaluating a model.
    """
    
    def __init__(self,
                 model_path,
                 tokenizer=None,
                 encoder_model=None):
        """
        #TODO update the docstring
        Initialization function for the ParagraphSelector class

        :param model_path: path to an already trained model (only
                           necessary if we want to load a pretrained
                           model)
        :param tokenizer: a tokenizer, default is BertTokenizer.from_pretrained('bert-base-uncased')
        :param encoder_model: an encoder model, default is BertModel.from_pretrained('bert-base-uncased')
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') if not tokenizer else tokenizer

        class ParagraphSelectorNet(BertPreTrainedModel):
            """
            A neural network for the paragraph selector.
            """
            def __init__(self, config):#, input_size=768, output_size=1):
                """
                #TODO update the docstring
                Initialization of the encoder model and a linear layer

                :param intput_size: input size for the linear layer
                :param output_size: output size of the linear layer
                """
                super(ParagraphSelectorNet, self).__init__(config)
                self.bert = BertModel(config)#('bert-base-uncased',
                                                               #output_hidden_states=True,
                                                               #output_attentions=True) if not encoder_model else encoder_model
                self.linear = torch.nn.Linear(config.hidden_size, 1)
                self.init_weights()

            def forward(self, token_ids):
                """
                #TODO update the docstring
                Forward function of the ParagraphSelectorNet.
                Takes in token_ids corresponding to a query+paragraph
                and returns a relevance score (between 0 and 1) for
                the query and paragraph.

                :param token_ids: token_ids as returned by the tokenizer;
                                  the text that is passed to the tokenizer
                                  is constructed by [CLS] + query + [SEP] + paragraph + [SEP]
                """

                # [-2] is all_hidden_states
                # [-1] is the last hidden state (list of sentences)
                # [:,0,:] - we want for all the sentence (:),
                # only the first token (0) (this is the [CLS token]), 
                # all its dimensions (:) (768 with bert-base-uncased)

                #with torch.no_grad(): #TODO de-activate this?
                #embedding = self.bert(token_ids)[-2][-1][:, 0, :] #TODO maybe, this throws errors. in this case, look at Stalin's version below

                outputs = self.bert(token_ids)
                embedding = outputs[0][:, 0, :]

                output = self.linear(embedding)
                output = torch.sigmoid(output)
                return output
        
        # initialise a paragraph selector net and try to load


        self.config = BertConfig.from_pretrained(model_path)  # , cache_dir=args.cache_dir if args.cache_dir else None,)
        self.net = ParagraphSelectorNet.from_pretrained(model_path,
                                                        from_tf=bool(".ckpt" in model_path),
                                                        config=self.config)  # , cache_dir=args.cache_dir if args.cache_dir else None,)

        '''
        self.net = ParagraphSelectorNet(self.config)
        if model_path:
            try:
                self.net.load_state_dict(torch.load(model_path))
            except FileNotFoundError as e:
                print(e, model_path)
        '''

    def train(self, train_data, dev_data, model_save_path,
              epochs=10, batch_size=1, learning_rate=0.0001, eval_interval=None, try_gpu=True):
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
        :return dev_scores: a list of tuples (evaluation step, p, r, f1, acc.)
        """
        # Use Binary Cross Entropy as a loss function instead of MSE
        # There are papers on why MSE is bad for classification
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        losses = []
        dev_scores = []

        # Set the network into train mode
        self.net.train()
        
        
        cuda_is_available = torch.cuda.is_available() if try_gpu else False
        device = torch.device('cuda') if cuda_is_available else torch.device('cpu')

        # put the net on the GPU if possible
        if cuda_is_available:
            #print("in PS.train(): cuda IS available!")#CLEANUP
            self.net = self.net.to(device)
        #else:
            #print("in PS.train(): cuda NOT available!")#CLEANUP

        print("Training...")
        
        #TODO: find a way to shuffle reproducibly

        # (120, 250) --> batching --> (30, 4, 250)
        train_data = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True)
        #print(f"in PS.train: shape of train_data: {len(train_data.dataset)}")  # CLEANUP

        c = 0  # counter over taining examples
        high_score = 0
        eval_interval = eval_interval if eval_interval else float('inf')
        batched_interval = round(eval_interval/batch_size) # number of batches needed to reach eval_interval
        a_model_was_saved_at_some_point = False

        for epoch in range(epochs):
            print('Epoch %d/%d' % (epoch + 1, epochs))

            for step, batch in enumerate(tqdm(train_data, desc="Iteration")):
                batch = [t.to(device) if t is not None else None for t in batch]
                inputs, labels = batch
                #weight_tensor = torch.Tensor([WEIGHTS[int(label)] for label in labels]).to(device) #CLEANUP?
                #criterion.weight = weight_tensor #CLEANUP?
                #print(inputs.shape) #CLEANUP

                optimizer.zero_grad()

                outputs = self.net(inputs).squeeze(1) #TODO why squeeze(1)?
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                losses.append(loss.item())

                c +=1
                # Evaluate on validation set after some iterations
                if c % batched_interval == 0:
                    p, r, f1, accuracy, _, _, _ = self.evaluate(dev_data, try_gpu=try_gpu)
                    dev_scores.append((c/batched_interval, p, r, f1, accuracy))

                    measure = f1
                    if measure > high_score:
                        print(f"Better eval found with score {round(measure ,3)} (+{round(measure-high_score, 3)})")
                        high_score = measure
                        self.net.save_pretrained(model_save_path)
                        a_model_was_saved_at_some_point = True
                    else:
                        print(f"No improvement yet...")



                optimizer.step()

        if not a_model_was_saved_at_some_point: # make sure that there is a model file
            self.net.save_pretrained(model_save_path)

        return losses, dev_scores
    
    def evaluate(self, data, threshold=0.1, text_length=512, try_gpu=True):
        """
        #TODO mention here that it makes labels on the fly (per question)
        Evaluate a trained model on a dataset.

        :param data: a list of datapoints where each point has the
                     following structure:
                        (question_id, supporting_facts, query, paragraphs),
                        where question_id is a string corresponding
                              to the datapoint id in HotPotQA
                        supporting_facts is a list of strings,
                        query is a string,
                        paragraphs is a 10-element list where
                            the first element is a string
                            the second element is a list of sentences (i.e., a list of strings)

        :param threshold: a float between zero and one;
                          paragraphs that get a score above the
                          threshold, become part of the context,
                          default is 0.1
        :param text_length: text_length of the paragraph - paragraph will
                            be padded if its lenght is less than this value
                            and trimmed if it is more, default is 512
        :param try_gpu: boolean specifying whether to use GPU for
                        computation if GPU is available; default is True

        :return precision: precision for the model
        :return recall: recall for the model
        :return f1: f1 score for the model
        :return acc: accuracy for the model
        :return ids: list of ids of all the evaluated points
        :return all_true: true labels for the datapoints
                          list(list(boolean)), a list of datapoints
                          where each datapoint is a list of 
                          booleans; each boolean corresponds to whether
                          the corresponding paragraph is relevant to
                          the query or not
        :return all_pred: precited labels for the datapoints
                          list(list(boolean)), a list of datapoints
                          where each datapoint is a list of 
                          booleans; each boolean corresponds to whether
                          the corresponding paragraph is relevant to
                          the query or not
        """
        all_true = []
        all_pred = []
        ids = []

        # TODO maybe put the model onto the GPU here instead of in make_context?
        self.net.eval()
        device = torch.device('cuda') if try_gpu and torch.cuda.is_available() \
            else torch.device('cpu')
        self.net = self.net.to(device)

        for point in tqdm(data, desc="eval points"):
            context = self.make_context(point,
                                        threshold=threshold,
                                        text_length=text_length,
                                        device=device) #point[2] are the paragraphs, point[1] is the query
            para_true = []
            para_pred = []
            for para in point[3]: # iterate over all 10 paragraphs
                para_true.append(para[0] in point[1]) # true if paragraph's title is in the supporting facts
                para_pred.append(para in context)
            all_true.extend(para_true)
            all_pred.extend(para_pred)
            ids.append(point[0])
            #print(f"in evaluate(): predicted: {para_true}") #CLEANUP
            #print(f"                    true: {para_pred}\n")

        
        precision = precision_score(all_true, all_pred)
        recall = recall_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred)
        acc = accuracy_score(all_true, all_pred)
        
        return precision, recall, f1, acc, ids, all_true, all_pred
    
    def predict(self, p, device=torch.device('cpu')):
        """
        Given the token_ids of a query+paragraph for a specific paragraph,
        return the relevance score that the model predicts between the query
        and the paragraph

        :param p: token_ids as returned by the tokenizer;
                  the text that is passed to the tokenizer
                  is constructed by [CLS] + query + [SEP] + paragraph + [SEP]
        :return: score between 0 and 1 for that paragraph
        """

        # put the net and the paragraph onto the GPU if possible
        #cuda_is_available = torch.cuda.is_available()
        #device = torch.device('cuda') if cuda_is_available else torch.device('cpu')
        #if cuda_is_available:
        #    self.net = self.net.to(device)
        #    p = p.to(device) #CLEANUP?
        #self.net.eval() #CLEANUP?


        p = p.to(device)
        score = self.net(p)
        return score
    
    def make_context(self, datapoint, threshold=0.1,
                     context_length=512, text_length=512,
                     device=torch.device('cpu')):
        """
        Given a datapoint from HotPotQA, build the context for it.
        The context consists of all paragraphs included in that
        datapoint which have a relevance score higher than a 
        specific value (threshold) to the query of that datapoint.
         
        :param datapoint: datapoint for which to make context
                          shape: (question_id, supporting_facts, query, paragraphs, answer),
                                where question_id is a string corresponding
                                      to the datapoint id in HotPotQA
                                supporting_facts is a list of strings,
                                query is a string,
                                paragraphs is a 10-element list of lists where
                                    the first element is a string
                                    the second element is a list of sentences (i.e., a list of strings)
        :param threshold: a float between zero and one;
                          paragraphs that get a score above the
                          threshold, become part of the context,
                          default is 0.1
        :param text_length: text_length of the paragraph - paragraph will #TODO update this: paragraph-individual trimming
                            be padded if its length is less than this value
                            and trimmed if it is more, default is 512
        :param device: device for processing; default is 'cpu'

        :return context: the context for the datapoint (title and paragraph are ids) # TODO downdate this!
                shape: [ [[p1_title], [p1_s1, p1_s2, ...]],
                         [[p2_title], [p2_s1, p2_s2, ...]],
                        ...]

        """

        # for the case that a user picks a limit greater than BERT's max length
        if text_length > 512:
            print("Maximum input length for Paragraph Selector exceeded; continuing with 512.")
            text_length = 512
        if context_length > 512:
            print("Maximum context length exceeded; continuing with 512.")
            context_length = 512


        context = []

        # encode header and paragraph individually to be able to join just paragraphs
        # automatically prefixes [CLS] and appends [SEP]
        query_token_ids = self.tokenizer.encode(datapoint[2],
                                                 max_length=512) # to avoid warnings
        """ SELECT PARAGRAPHS """
        for p in datapoint[3]:
            header_token_ids = self.tokenizer.encode(p[0],
                                                   max_length=512, # to avoid warnings
                                                   add_special_tokens=False)
            # encode sentences individually
            sentence_token_ids = [self.tokenizer.encode(sentence,
                                                   max_length=512, # to avoid warnings
                                                   add_special_tokens=False)
                              for sentence in p[1]]

            token_ids = query_token_ids \
                      + header_token_ids \
                      + [token for sent in sentence_token_ids for token in sent]
            token_ids[-1] = self.tokenizer.sep_token_id  # make sure that it ends with a SEP

            # Add padding if there are fewer than text_length tokens,
            if len(token_ids) < text_length:
                token_ids += [self.tokenizer.pad_token_id for _ in range(text_length - len(token_ids))]
            else: # else trim to text_length
                token_ids = token_ids[:text_length]
                token_ids[-1] = self.tokenizer.sep_token_id  # make sure that it still ends with a SEP

            # do the actual prediction & decision
            encoded_p = torch.tensor([token_ids])
            score = self.predict(encoded_p, device=device)
            if score > threshold:
                # list[list[int], list[list[int]]]
                # no [CLS] or [SEP] here
                context.append([header_token_ids, sentence_token_ids])

        """ TRIM EACH PARAGRAPH OF THE CONTEXT """
        # shorten each paragraph so that the combined length is not too big
        # and decode so that strings are returned
        #TODO maybe extract this to a function
        trimmed_context = []# new data structure because we prioritise computing time over memory usage

        cut_off_point = 0 if not context else math.ceil(context_length/len(context)) # roughly cut to an even length
        #print(f"======== PARAGRAPH SELECTOR: SHORTENING THE DAMN THING ========") #CLEANUP
        #print(f"cut_off_point: {cut_off_point}") #CLEANUP
        for i, (header, para) in enumerate(context):

            if len(header) >= cut_off_point:
                trimmed_context.append([self.tokenizer.decode(header[:cut_off_point]), []])
                #print(f"paragraph {i}: header is longer than cut_off_point! ({len(header)} vs. {cut_off_point})") #CLEANUP
                continue # don't even look at the paragraph
            else:
                pos = len(header) # the header counts towards the paragraph!
                trimmed_context.append([self.tokenizer.decode(header), []])
                #print(f"paragraph {i}, pos: {pos}")  # CLEANUP

            for sentence in para:
                #print(f"   pos: {pos}   cut-off point: {cut_off_point}   tokens: {len(sentence)}") #CLEANUP
                if pos + len(sentence) > cut_off_point:
                    s = sentence[:cut_off_point - pos] # trim
                    #print(f"   cut \n   {len(sentence)} {sentence}\n   to\n   {len(s)} {s}\n") #CLEANUP
                    s = self.tokenizer.decode(s) # re-convert to a string
                    if len(s) != 0:
                        trimmed_context[i][1].append(s)
                    break # don't continue to loop over further sentences of this paragraph
                else:
                    s = self.tokenizer.decode(sentence)
                    trimmed_context[i][1].append(s) # append non-trimmed sentence to the context
                    pos += len(sentence) # go to the next sentence

        #print(f"roughly trimmed context:") #CLEANUP
        #pprint(trimmed_context) #CLEANUP
        #print("\n\n") #CLEANUP

        return trimmed_context

    def save(self, savepath):
        '''
        Save the trained model to a file.

        :param savepath: relative path to where the model
                         should be saved, including filename(?)
        '''
        directory_name = "/".join(savepath.split('/')[:-1])
        print("Save to:", directory_name)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        torch.save(self.net.state_dict(), savepath)

if __name__ == "__main__":
    #TODO update this to match train_ps.py
    timer = Timer()

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', metavar='config', type=str,
                        help='configuration file for training')
    args = parser.parse_args()

    cfg = ConfigReader(args.config_file)

    dataset_size = cfg("dataset_size")
    test_split = cfg("test_split")
    shuffle_seed = cfg("shuffle_seed")
    text_length = cfg("text_length") # limits paragraph length in order to reduce complexity

    epochs = cfg("epochs")
    batch_size = cfg("batch_size")
    learning_rate = cfg("learning_rate")

    training_data_rel_path = cfg("training_data_rel_path")
    model_rel_path =  cfg("model_rel_path")
    losses_rel_path = cfg("losses_rel_path")
    predictions_rel_path = cfg("predictions_rel_path")
    performance_rel_path = cfg("performance_rel_path")


    print("Reading data...")
    dh = HotPotDataHandler(parent_dir + training_data_rel_path)
    data = dh.data_for_paragraph_selector()
    timer("data_input")

    print("Splitting data...")
    training_data_raw, test_data_raw = train_test_split(data[:dataset_size],
                                                        test_size=test_split,
                                                        random_state=shuffle_seed,
                                                        shuffle=True)
    train_tensor = make_training_data(training_data_raw, text_length=text_length)
    timer("data_splitting")

    print("Initilising ParagraphSelector...")
    ps = ParagraphSelector()
    losses = ps.train(train_tensor,
                      epochs=epochs,
                      batch_size=batch_size,
                      learning_rate=learning_rate)
    timer("training")

    print("Saving model and losses...")
    ps.save(parent_dir + model_rel_path)
    with open(parent_dir + losses_rel_path, "w") as f:
        f.write("\n".join([str(l) for l in losses]))
    timer("saving_model")


    print("Evaluating...")
    precision, recall, f1, ids, y_true, y_pred = ps.evaluate(test_data_raw,
                                                             text_length=text_length,
                                                             try_gpu=True)
    print('----------------------')
    print("Precision:", precision)
    print("Recall:", recall)
    print("F score:", f1)
    timer("evaluation")

    if not os.path.exists(parent_dir + "/models/performance/"):
        os.makedirs(parent_dir + "/models/performance/")

    with open(parent_dir + predictions_rel_path, 'w', encoding='utf-8') as f:
        for i in range(len(ids)):
            f.write(ids[i] + "\t" + \
                    ','.join([str(int(j)) for j in y_true[i]]) + "\t" + \
                    ','.join([str(int(j)) for j in y_pred[i]]) + "\n")

    with open(parent_dir + performance_rel_path, 'w', encoding='utf-8') as f:
        f.write("Configuration in: " + args.config_file + "\n")
        f.write("Outputs in:  " + parent_dir + predictions_rel_path + \
                "\nPrecision: " + str(precision) + \
                "\nRecall:    " + str(recall) + \
                "\nF score:   " + str(f1) + "\n")
        f.write("Hyper parameters:" + \
                "\ndataset size: " + str(dataset_size) + \
                "\ntest split:   " + str(test_split) + \
                "\nshuffle seed: " + str(shuffle_seed) + \
                "\ntext length:  " + str(text_length) + \
                "\nepochs:       " + str(epochs) + \
                "\nbatch size:   " + str(batch_size))

        timer.total()
        f.write("\n\nTimes taken:\n" + str(timer))
        print("\ntimes taken:\n", timer)



"""
This module implements the Paragraph Selector from the paper, Section 3.1
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os,sys,inspect
from tqdm import tqdm
import argparse

from sklearn.metrics import recall_score, precision_score, f1_score
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import HotPotDataHandler
from utils import ConfigReader
from utils import Timer


def make_training_data(data,
                       text_length=512,
                       pad_token_id = 0,
                       tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
    """
    Make a train tensor for each datapoint.

    :param data: question ID, supporting facts, question, and paragraphs, 
                 as returned by HotPotDataHandler
    :type data: list(tuple(str, list(str), str, list(list(str, list(str)))))
    :param text_length: id of the pad token used to pad when
                        the paragraph is shorter then text_length
                        default is 0
    :param pad_token_id: text_length of the paragraph - paragraph will
                         be padded if its lenght is less than this value
                         and trimmed if it is more, default is 512
    :param tokenizer: default: BertTokenizer(bert-base-uncased)

    :return: a train tensor with two columns:
                1. token_ids as returned by the tokenizer for
                   [CLS] + query + [SEP] + paragraph + [SEP]
                   (10 entries per datapoint, one of each paragraph)
                2. labels for the points - 0 if the paragraphs is
                   no relevant to the query, and 1 otherwise
    """

    labels = []
    datapoints = []
    for point in tqdm(data):
        for para in point[3]:
            # Label is 1: if paragraph title is in supporting facts, otherwise 0
            labels.append(float(para[0] in point[1]))
            point_string = point[2] + " [SEP] " + ("").join(para[1])
            
            # automatically prefixes [CLS] and appends [SEP]
            token_ids = tokenizer.encode(point_string, max_length=512)
            
            # Add padding if there are fewer than text_length tokens,
            # else trim to text_length
            if len(token_ids) < text_length:
                token_ids += [pad_token_id for _ in range(text_length - len(token_ids))]
            else:
                token_ids = token_ids[:text_length]
            datapoints.append(token_ids)
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
                 model_path=None,
                 tokenizer=None,
                 encoder_model=None):
        """
        Initialization function for the ParagraphSelector class

        :param model_path: path to an already trained model (only
                           necessary if we want to load a pretrained
                           model)
        :param tokenizer: a tokenizer, default is BertTokenizer.from_pretrained('bert-base-uncased')
        :param encoder_model: an encoder model, default is BertModel.from_pretrained('bert-base-uncased')
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') if not tokenizer else tokenizer

        class ParagraphSelectorNet(torch.nn.Module):
            """
            A neural network for the paragraph selector.
            """
            def __init__(self, input_size=768, output_size=1):
                """
                Initialization of the encoder model and a linear layer

                :param intput_size: input size for the linear layer
                :param output_size: output size of the linear layer
                """
                super(ParagraphSelectorNet, self).__init__()
                self.encoder_model = BertModel.from_pretrained('bert-base-uncased',
                                                               output_hidden_states=True,
                                                               output_attentions=True) if not encoder_model else encoder_model
                self.linear  = torch.nn.Linear(input_size, output_size)

            def forward(self, token_ids):
                """
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

                with torch.no_grad(): #TODO de-activate this?
                    embedding = self.encoder_model(token_ids)[-2][-1][:, 0, :]

                output = self.linear(embedding)
                output = torch.sigmoid(output)
                return output 
        
        # initialise a paragraph selector net and try to load
        # a trained model from a file, if a file has been specified
        self.net = ParagraphSelectorNet()
        if model_path:
            try:
                self.net.load_state_dict(torch.load(model_path))
            except FileNotFoundError as e:
                print(e, model_path)

    def train(self, train_data, epochs=10, batch_size=1, learning_rate=0.0001):
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
        
        #TODO: find a way to shuffle reproducibly
        train_data = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True) 

        # Iterate over the epochs
        for epoch in range(epochs):
            print('Epoch %d/%d' % (epoch + 1, epochs))
            
            for step, batch in enumerate(tqdm(train_data, desc="Iteration")):
                batch = [t.to(device) if t is not None else None for t in batch]
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = self.net(inputs).squeeze(1) #TODO why squeeze(1)?
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                losses.append(loss.item())
                optimizer.step()

        return losses
    
    def evaluate(self, data, threshold=0.1, pad_token_id=0, text_length=512, try_gpu=True):
        """
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
        :param pad_token_id: id of the pad token used to pad when
                             the paragraph is shorter then text_length
                             default is 0
        :param text_length: text_length of the paragraph - paragraph will
                            be padded if its lenght is less than this value
                            and trimmed if it is more, default is 512
        :param try_gpu: boolean specifying whether to use GPU for
                        computation if GPU is available; default is True

        :return precision: precision for the model
        :return recall: recall for the model
        :return f1: f1 score for the model
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

        for point in tqdm(data, desc="Datapoints"):
            context = self.make_context(point,
                                        threshold=threshold,
                                        pad_token_id=pad_token_id,
                                        text_length=text_length,
                                        device=device) #point[2] are the paragraphs, point[1] is the query
            para_true = []
            para_pred = []
            for para in point[3]:
                para_true.append(para[0] in point[1])
                para_pred.append(para in context)
            all_true.append(para_true)
            all_pred.append(para_pred)
            ids.append(point[0])
        
        # Flatten the lists so they can be passed to the precision and recall funtions
        all_true_flattened = [point for para in all_true for point in para]
        all_pred_flattened = [point for para in all_pred for point in para]
        
        precision = precision_score(all_true_flattened, all_pred_flattened)
        recall = recall_score(all_true_flattened, all_pred_flattened)
        f1 = f1_score(all_true_flattened, all_pred_flattened)
        
        return precision, recall, f1, ids, all_true, all_pred
    
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
                     pad_token_id=0, text_length=512, device=torch.device('cpu')):
        """
        Given a datapoint from HotPotQA, build the context for it.
        The context consists of all paragraphs included in that
        datapoint which have a relevance score higher than a 
        specific value (threshold) to the query of that datapoint.
         
        :param datapoint: datapoint for which to make context
                          shape: (question_id, supporting_facts, query, paragraphs),
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
        :param pad_token_id: id of the pad token used to pad when
                             the paragraph is shorter then text_length
                             default is 0
        :param text_length: text_length of the paragraph - paragraph will
                            be padded if its lenght is less than this value
                            and trimmed if it is more, default is 512
        :param device: device for processing; default is 'cpu'

        :return context: the context for the datapoint
                shape: [[p1_title, [p1_s1, p1_s2 ...]],
                        [p2_title, [p2_s1, p2_s2, ...]],
                        ...]

        """
        context = []

        for p in datapoint[3]:
            # automatically prefixes [CLS] and appends [SEP]
            token_ids = self.tokenizer.encode(datapoint[2] + " [SEP] " + ("").join(p[1]),
                                              max_length=512)

            # Add padding if there are fewer than text_length tokens,
            # else trim to text_length
            if len(token_ids) < text_length:
                token_ids += [pad_token_id for _ in range(text_length - len(token_ids))]
            else:
                token_ids = token_ids[:text_length]

            # do the actual prediction & decision
            encoded_p = torch.tensor([token_ids])
            score = self.predict(encoded_p, device=device)
            if score > threshold:
                context.append(p)

        return context

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



"""
This module implements the Paragraph Selector from the paper, Section 3.1
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from utils import HotPotDataHandler

def encode(text,
           tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
           model=BertModel.from_pretrained('bert-base-uncased',
                                 output_hidden_states=True,
                                 output_attentions=True)):
    ''' TODO: document
    '''
    
    input_ids = torch.tensor([tokenizer.encode(text)])
    all_hidden_states, all_attentions = model(input_ids)[-2:]
    
    # This is the embedding of the [CLS] token.
    # [-1] is the last hidden state (list of sentences)
    # first [0] - first (and only) sentence
    # second [0] - first ([CLS]) token of the sentence
    return all_hidden_states[-1][0][0]

def make_training_data(data,
                       tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
                       model=BertModel.from_pretrained('bert-base-uncased',
                                 output_hidden_states=True,
                                 output_attentions=True)):
    '''
    Make a dataframe with training data for selecting relevant paragraphs
    Each entry in the dataframe has three columns:
        1. Query - the question
        2. Paragraphs - the paragraphs
        3. Label - 0 (unrelated) or 1 (related)
    '''
    labels = []
    datapoints = []
    for point in data:        
        for para in point[2]:
            labels.append(int(para[0] in point[0])) # Label 1: if paragraph title is in supporting facts, otherwise 0
            encoded_point = encode("[CLS] " + point[1] + " [SEP] " + ("").join(para[1]) + " [SEP]", tokenizer, model)
            datapoints.append(encoded_point)
        
    df = pd.DataFrame({
        'id': range(len(labels)),
        'label': labels,
        'text': datapoints
    })
    return df  

class ParagraphSelector():
    """
    TODO: write docstring
    """
    
    def __init__(self,
                 model_path=None,
                 tokenizer=None,
                 encoder_model=None):
        """
        TODO: write docstring
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') if not tokenizer else tokenizer
        self.encoder_model = BertModel.from_pretrained('bert-base-uncased',
                                 output_hidden_states=True,
                                 output_attentions=True) if not encoder_model else encoder_model
        
        class ParagraphSelectorNet(torch.nn.Module):
            """
            TODO: write docstring
            """
            def __init__(self, input_size=768, output_size=1):
                super(ParagraphSelectorNet, self).__init__()
                self.linear  = torch.nn.Linear(input_size, output_size)

            def forward(self, embedding):
                output = self.linear(embedding)
                output = torch.sigmoid(output)

                return output 
            
        self.net = ParagraphSelectorNet()
        if model_path:
            try:
                self.net.load_state_dict(torch.load(model_path))
            except FileNotFoundError as e:
                print(e, model_path)
    
    def encode(text):
        ''' TODO: document
        '''

        input_ids = torch.tensor([self.tokenizer.encode(text)])
        all_hidden_states, all_attentions = self.encoder_model(input_ids)[-2:]

        # This is the embedding of the [CLS] token.
        # [-1] is the last hidden state (list of sentences)
        # first [0] - first (and only) sentence
        # second [0] - first ([CLS]) token of the sentence
        return all_hidden_states[-1][0][0]

    def train(self, train_data, labels, epochs, learning_rate=0.0001):
        """
        TODO: write docstring
        """
        # Use Binary Cross Entropy as a loss function instead of MSE
        # There are papers on why MSE is bad for classification
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        losses = []
        
        # Set the network into train mode
        self.net.train()

        print("Training...")

        # Iterate over the epochs
        for epoch in range(epochs):
            print('Epoch %d/%d' % (epoch + 1, epochs))
            for inputs, label in zip(train_data, labels):

                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, label)
                loss.backward(retain_graph=True)
                losses.append(loss.item())
                optimizer.step()

        return losses
    
    def evaluate(self, data, threshold=0.1):
        """
        TODO: write docstring
        """
        y_true = []
        y_pred = []
        
        for point in data:
            context = self.make_context(point, threshold) #point[2] are the paragrphs, point[1] is the query
            for para in point[2]:
                y_true.append(para[0] in point[0])
                y_pred.append(para in context)
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        return precision, recall
    
    def predict(self, p):
        """ Given the encoding of a paragraph (query + paragraph),
        return the score that the model predicts for that paragraph
        
        Parameters: p - encoding of the paragraph (along with the query),
                        as described in the paper
        Output: score between 0 and 1 for that paragraph
        """
        self.net.eval()
        score = self.net(p)
        return score
    
    def make_context(self, datapoint, threshold=0.1):
        """
        TODO: write docstring
        
        Parameters: paragraphs - [[p1_title, [p1_s1, p1_s2 ...]],
                                  [p2_title, [p2_s1, p2_s2, ...]],
                                   ...]
                    query - the query as a string
                    threshold - a float between zero and one;
                                paragraphs that get a score above the
                                threshold, become part of the context
        Output: context: [[p1_title, [p1_s1, p1_s2 ...]],
                          [p2_title, [p2_s1, p2_s2, ...]],
                           ...]
        """
        context = []
        for p in datapoint[2]:
            # p[0] is the paragraph title, p[1] is the list of sentences in the paragraph
            encoded_p = self.encode("[CLS] " + datapoint[1] + " [SEP] " + ("").join(p[1]) + " [SEP]")
            score = self.predict(encoded_p)
            if score > threshold:
                context.append(p)
        return context
    
    def save(self, savepath):
        '''
        TODO: write docstring
        '''
        directory_name = "/".join(savepath.split('/')[:-1])
        print(directory_name)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        torch.save(self.net.state_dict(), savepath)

if __name__ == "__main__":
    print("Reading data...")
    dh = HotPotDataHandler(parent_dir + "/data/hotpot_train_v1.1.json")
    data = dh.data_for_paragraph_selector()
    
    print("Splitting data...")
    training_data_raw, test_data_raw = train_test_split(data[:4], test_size=0.25, random_state=42, shuffle=True)
    training_data = make_training_data(training_data_raw)
    training_data = shuffle(training_data, random_state=42)
    
    print("Initilising ParagraphSelector...")
    ps = ParagraphSelector()
    losses = ps.train(training_data["text"], training_data["label"], 1)
    
    print("Saving model...")
    ps.save(parent_dir + '/models/paragraphSelector_all.pt')
    
    print("Evaluating...")
    precision, recall = ps.evaluate(test_data_raw)
    print('----------------------')
    print("Precision:", precision)
    print("Recall:", recall)
    
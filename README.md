# Advances_in_QA: Re-implementing 'Dynamically Fused Graph Networks'

Here are exercises and the final project from the "Advances in QA" class, winter semester 2019/2020, Uni Saarland. 

The final project aims to re-implement the system published by [Xiao et al. (2019)](https://arxiv.org/pdf/1905.06933.pdf "pdf on arxiv.org"), *"Dynamically Fused Graph Network for Multi-hop Reasoning"* with as little assistance from their [openly available code](https://github.com/woshiyyya/DFGN-pytorch "Github repository") as possible.

### DFGN in Short
Multi-hop question answering (QA) requires a system to derive the answer to a question from multiple text resources which, each on its own, don't contain the full answer.

In short, the workflow of dynamically fused graph networks (DFGN) is as follows: select several relevant paragraphs, construct an entity graph from them, and then look at parts of this graph to compute which entities contribute the most to answering the question at hand. Perform this step multiple times (each time looking at different parts of the entity graph) and take into account the entities' contribution from the previous iteration. This way, the graph network converges to the final answer.

The paper describes an architecture which is split into 5 modules:
1) **paragraph selector** returns only the most relevant paragraphs (= "the context")
2) **graph constructor** returns an entity graph from the context
3) **encoder** uses BERT and BiDAF to encode the context and the question
4) **fusion block** – the heart of DFGN – looks at parts of the entity graph for multiple iterations and exchanges information between the graph's nodes
5) **predictior** takes the fusion block's output and passes it through a stacked LSTM architecture to output the final answers

The intuition is that with multiple iterations, relevant entities propagate their importance to other, directly connected entities. The entity graph has no document boundaries, enabling free flow of information ( = reasoning) across paragraphs. By focusing on sub-parts of the entity graph, **[TODO] why only look at subgraphs?** 


### External and Internal Modules
- `utils.py` (local) provides helper functions and classes.
- `torch` [(get it here)](https://pytorch.org/)
- `tqdm` for progress bars. [(get it here)](https://tqdm.github.io/ "Github")
- `flair` for named entity recognition (NER) [(get it here)](https://github.com/flairNLP/flair). `Python 3.8` and above cause issues with `flair`, use a lower version. We have run this with `Python 3.6`
- `pycorenlp.StanfordCoreNLP` for NER (not necessary) [(get it here)](https://stanfordnlp.github.io/CoreNLP/other-languages.html)
- `transformers` by Huggingface, supplying BERT [(get it here)](https://github.com/huggingface/transformers#installation)
- `sklearn` mainly for evaluation [(get it here)](https://github.com/scikit-learn/scikit-learn)
- `ujson` for running the official HotPotQA evaluation script [(get it here)](https://github.com/ultrajson/ultrajson)
- `pandas` [(get it here)](https://github.com/pandas-dev/pandas)

You can install each of these modules individually or use the `requirements.txt` file:

```
pip install -r requirements.txt
```

### Train the Paragraph Selector with `train_ps.py`
Pass a configuration file and a model name for execution. The model name will be used to **create a directory with all outputs** (model config, model parameters, losses, times, scores during training). Example:
```
python3 train_ps.py config/train_ps_final.cfg my_model
```



### Train the DFGN with `train_dfgn.py`
Training a DFGN means that the Encoder, FusionBlock, and Predictor modules are trained jointly, using a ParagraphSelector model and the EntityGraph module to process a question before it is encoded. This script runs similarly to `train_ps.py`:
```
python3 train_dfgn.py config/train_dfgn.cfg my_dfgn
```
[TODO update this section?]
Have a look at the config file used in this example in oder to get an idea of the required (and optional) parameters for training. If you run into issues with your GPU, try setting device-related parameters to "False" or to 'cpu'. The batch size might have to be very small. 



### Test the Paragraph Selector with `eval_ps.py`
This is just as straightforward as training: upon execution, pass a configuration file and name of the model that you want to test to `eval_ps.py` and the script will compute precision, recall, F1 score, and accuracy and log them:
```
python3 eval_py.py config/eval_ps.cfg my_ParagraphSelector_model
```
The predictions made during evaluation are also logged in a directory named after the model. 



### Test the DFGN with `eval_dfgn.py` 
Similarly to the evaluation script for the Paragraph Selector, pass a configuration file and the name of the directory containing the model. 
```
python3 eval_dfgn.py config/eval_dfgn.cfg my_DFGN_model
```


### Pre-trained models [TODO make sure that this is up-to-date]
You can download pre-trained models for the ParagraphSelector and the subsequent DFGN [from this Google Drive](https://drive.google.com/drive/folders/1FZzxpKQGhDzaDjACcPTna117Ope-RKdE?usp=sharing).


### Configuration Files
The class `ConfigReader` in the utils module can parse files in raw text format to a number of data types. The general syntax of configuration files (preferably indicated by the extension '.cfg') follows Python syntax. Here are important details:
 
- one parameter per line, containing a name and a value
    - name and value are separated by at least one white space or tab
    - names should only contain alphanumeric symbols and '_' (no '-', please!)
- list-like values are allowed (use Python list syntax)
    - strings within value lists don't need to be quoted
    - value lists either with or without quotation (no `["foo", 3, "bar"]` )
    - mixed lists will exclude non-quoted elements
- multi-word strings are marked with single or double quotation marks
- strings containing quotation marks are not tested yet. Be careful!
- lines starting with `#` are ignored
- no in-line comments!
- config files should have the extension 'cfg' (to indicate their purpose)

Suppose there is a file called 'my_config.cfg' which contains a line `batch_size   42`. ConfigReader can be used to access the value 42 like this:
```python
from utils import ConfigReader
file_path = 'my_config.cfg'
cfg = ConfigReader(file_path)
my_batch_size = cfg("batch_size")
```

ConfigReader objects hold the parsed parameters as a dictionary, which allows to access all (or sets of) parameters at once.
Note that there is no control of whether all parameters that are required for the execution of a program are actually specified in the config file, and that ConfigReader returns `None` for parameters that it doesn't hold.


### Files and Directories

- `modules/` — the main modules of the architecture
    - `ParagraphSelector.py` - implements the Paragraph Selector from the paper (section 3.1)
    - `EntityGraph.py`- implements the Graph Constructor from the paper (section 3.2) and builds the binary matrix used in section 3.4
    - `Encoder.py` - implements the Encoder from the paper (section 3.3)
    - `FusionBlock.py` - implements the Fusion Block from the paper (section 3.4)
    - `Predictor.py` - implements the LSTM Prediction Layer from the paper (section 3.5)
- `config/` — configuration files; input to ConfigReader objects
- `models/` — results on performance tests of (ParagraphSelector, DFGN) models
- `playground/` — code snippets and little scripts; unimportant for running code

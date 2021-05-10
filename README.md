# TeMP: Temporal Message Passing Network for Temporal Knowledge Graph Completion
PyTorch implementation of [TeMP: Temporal Message Passing Network for Temporal Knowledge Graph Completion](https://arxiv.org/pdf/2010.03526.pdf) ([EMNLP 2020](https://2020.emnlp.org/))

**Update:** you can now download the available trained model [here](https://drive.google.com/file/d/1efPogR01qHorVX4lpqstMqyATENWVP6C/view?usp=sharing). The hyperparameter configuration are both suggested by the folder name, and detailed in the config.json in each checkpoint folder.
## Installation
Create a conda virtual environment first, you can name `your_env_name` yourself:
```
conda create --name <your_env_name> python=3.6.10
conda activate <your_env_name>
```

Assuming that you are using cuda 10.1, the package installation process is as follows:

```
conda install pytorch=1.3.0 cudatoolkit=10.1 -c pytorch && conda install -c dglteam dgl-cuda10.1==0.4.1 && python -m pip install -U matplotlib && pip install -r requirements.txt
```

## Training a model

The config files are stored in the `grid` folder. The structure of the folder looks like this:
```
grid
├── icews14
├── icews15
└── gdelt      
```

Each subfolder contains the following effective config files:
```
icews14
├── config_bigrrgcn.json # bidirectional GRU + RGCN
├-- config_bisargcn.json # bidirectional Transformer + RGCN
├-- config_grrgcn.json   # one-directional GRU + RGCN
├-- config_sargcn.json   # one-directional Transformer + RGCN
└-- config_srgcn.json    # RGCN only
```


The following command trains a model using the bidirectional GRU + RGCN model with frequency based gating. The config file following `-c` provide a set of parameters that overwrites the default parameters. 
```
python -u main.py -c configs/grid/icews15/config_bisargcn.json --rec-only-last-layer --use-time-embedding --post-ensemble
```

  `--n-gpu`: index of the gpus for usage, e.g. `--n-gpu 0 1 2` for using GPU indexed 0, 1 and 2.
 
  `--module`: model architecture:
 
  - baselines：`Static` for static KG embedding, `SRGCN` for static RGCN; `DE`, `Hyte`. 
  - `GRRGCN`: GRU + RGCN; `BiGRRGCN`: BiGRU + RGCN
  - `SARGCN`: Transformer + RGCN; `BiSARGCN`: BiTransformer + RGCN
    
  `--dataset` or `-d`: name of the dataset, `icews14`, `icews05-15` or `gdelt`
 
  `--config`: name of the config file.
 
  `--score-function`: decoding function. Choose among `TransE`, `distmult` and `complex`. Default: complex
 
  `--negative-rate`: number of negative samples per training instance. Note that for both object and subject we sample this amount of negative entities.
  
  `--max-nb-epochs`: maximum number of training epoches
  
  `--patience`: stop training after waiting for this number of epochs after model achieving the best performance on validation set
  
  `--n_bases`: number of blocks in each block-diagonal relation matrix. Used for RGCN representation
  
  `--num_pos_facts`: number of sampled facts to construct the training graph at each time step
  
  `--train-seq-len`: number of time steps preceding each time step `t`, from which historical facts are sampled. 
  For single directional models, the model uses this number of snapshots preceeding the current time step. 
  For bidirectional model (BiGRRGCN or BiSARGCN), the model uses this number of time steps both before and after the current time step. 
   
  `--test-seq-len`: same as `--train-seq-len`, except that it is used at the test time. 
  
  Flag arguments:
  
  `--post-ensemble`: use frequency based gating (see paper)
  
  `--impute`: use imputation (see paper)
  
  `--learnable-lambda`: learn the temperature lambda as a learnable parameter, as described in the paper
  
  `--rec-only-last-layer`: use recurrence only in the last RGCN layer. We find this to be the most effective hence include it in the paper.  
  
  `--random-dropout`: randomly drop half of edges in each historical and/or future time step
  
  `--debug`: only train the model using 0.1 percent of the data for the sanity check purpose
  
  `--fast_dev_run`: runs full iteration over everything to find bugs
   
  `--type1`: use type 1 GRU cell defined by the [wikipedia page](https://en.wikipedia.org/wiki/Gated_recurrent_unit) implemented by ourselves
  
## Testing and analysis
To test a model on the corresponding test set, run the following:
```
python -u test.py --checkpoint-path ${path-to-your-model-checkpoint}
```
To perform various link prediction analysis:
```
python link_prediction_analysis.py --checkpoint-path ${path-to-your-model-checkpoint}
```

To get the prediction of the TED classifier, run `python greedy_classifier.py` with the desired parameters. 

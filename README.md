# Orchard Dataset

This repository contains the code used for generating the Orchard Dataset, as seen in the
Multi-Hierarchical Reasoning in Sequences: State-of-the-Art Neural Sequence Models Fail To Generalize paper. The coode to train and test Transformers and Bi-directional LSTM models was adapted from  
[Fairseq](https://github.com/pytorch/fairseq).



## Software Requirements
Python 3.6, PyTorch 1.4 are required for the current codebase. Install apex to enable fp16 training.

## Steps

1. Install PyTorch and apex by running pip install -r requirements.txt 

2.  Generate Orchard

+  Generate Orchard-easy Dataset with MIN-MAX operators.
```python generate_tree.py --c 0 --mm --size 50 --dir /path_to_data/```
      
      +  Generate Orchard-hard Dataset with FIRST-LAST operators.
      ```python generate_tree.py --c 1.0 --fl --size 50 --dir /path_to_data/```

3.  Pre-process Dataset

  	+ Pre-process Dataset to generate translation dictionaries
      ```python preprocess.py --trainpref /path_to_data/train --validpref /path_to_data/valid --source-lang input --target-lang label --task translation --testpref /path_to_data --destdir /path_to_data```

4.  Train model

      + Train Transformer
      ```python train.py /path_to_data/ --save-dir /path_to_data/ --task translation --source-lang input --target-lang label --batch-size 128 --arch transformer --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt --fp16 --adam-betas '(0.9, 0.98)' --weight-decay 1.2e-6 --clip-norm 1. --dropout 0.3 --save-interval 50 --max-epoch 500```
      
      + Train LSTM
      ```python train.py data-orchard-mmc --save-dir data-orchard-mmc --task translation --arch lstm  --source-lang input --target-lang label --batch-size 128 --save-interval 100 --max-epoch 500 --lr 5e-3 --fp16```
      
5.  Generate predictions

      + Test model on depth of tree 7
      ```python generate.py /path_to_data/test7 --path /path_to_data/checkpoint500.pt --batch-size 32 --beam 5```
      
    


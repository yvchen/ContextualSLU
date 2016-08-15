# ContextualSLU: Multi-Turn Spoken/Natural Language Understanding


*A Keras implementation of the models described in [Chen et al. (2016)] (https://www.csie.ntu.edu.tw/~yvchen/doc/IS16_ContextualSLU.pdf).*

This model implements a memory network architecture for multi-turn understanding, 
where the history utterances are encoded as vectors and stored into memory cells for the current utterance's attention to improve slot tagging.

## Content
* [Requirements](#requirements)
* [Getting Started](#getting-started)
* [Model Running](#model-running)
* [Contact](#contact)
* [Reference](#reference)

## Requirements
1. Python
2. Numpy `pip install numpy`
3. Keras and associated Theano or TensorFlow `pip install keras`
4. H5py `pip install h5py`

## Dataset
1. Train/Test: word sequences with IOB slot tags and the indicator of the dialogue start point (1: starting point; 0: otherwise) `data/cortana.communication.5.[train/dev/test].iob`


## Getting Started
You can train and test JointSLU with the following commands:

```shell
  git clone --recursive https://github.com/yvchen/ContextualSLU.git
  cd ContextualSLU
```
You can run a sample tutorial with this command:
```shell
  bash script/run_sample.sh memn2n-c-gru theano 0 | sh
```
Then you can see the predicted result in `sample/rnn+emb_H-100_O-adam_A-tanh_WR-embedding.test.3`.

## Model Running
To reproduce the work described in the paper.
You can run the baseline slot filling w/o contextual information using GRU by:
```shell
  bash script/run_sample.sh gru theano 0 | sh
```

## Contact
Yun-Nung (Vivian) Chen, y.v.chen@ieee.org

## Reference

Main papers to be cited
```
@Inproceedings{chen2016end,
  author    = {Chen, Yun-Nung and Hakkani-Tur, Dilek and Tur, Gokhan and Gao, Jianfeng and Deng, Li},
  title     = {End-to-End Memory Networks with Knowledge Carryover for Multi-Turn Spoken Language Understanding},
  booktitle = {Proceedings of Interspeech},
  year      = {2016}
}



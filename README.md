# Autoregressive Generation Strategies for Long-term Sequential Recommendations

This repository contains code for the paper ''**Autoregressive generation strategies for long-term sequential recommendations**''.

The goal of sequential recommendations is usually formulated in terms of next-item prediction. In this paper, we consider Transformer-based models for longer-term sequential recommendations. We introduce a generative approach to this task, which consists of training an autoregressive neural network on a classic next-item prediction task and autoregressive generation of recommendations item-by-item for the next N items. We explore different autoregressive generation strategies, including greedy decoding, beam search, and temperature sampling. In addition, we propose novel generation strategies based on multi-sequence generation with temperature sampling and subsequent aggregation. Experiments on diverse datasets with the GPT-2 model show that this approach can improve performance on longer time horizons.

## Usage

Install requirements:
```sh
pip install -r requirements.txt
```
Note, that the code doesn't work with `python>=3.10`, since `recommenders` library needs `python<3.10`.

`notebooks` contains code for baselines mentioned in the paper: BPR-MF, BERT4Rec and SASRec.

We use [Hydra](https://hydra.cc/) for configuring the experements and `ClearML` to log results.
All configurable parameters can be found in corresponding configs at `src/configs`, and also they can be overriden from the command line.

Below are examples of training GPT-2 and testing obtained model with different strategies.

### Train Model

```
python src/run_train.py --config-name=GPT_train data_path=data/ml-1m.csv task_name=ml1_GPT_train seqrec_module.lr=1e-3
```
To use trained model for inference, copy task ID `<TRAIN_ID>` from the obtained ClearML training page.

### Test Model

Top-K prediction:
```
python src/run_predict.py --config-name=GPT_predict task_name=ml1_GPT_predict train_task=<TRAIN_ID>
```
Greedy decoding:

```
python src/run_predict.py --config-name=GPT_greedy task_name=ml1_GPT_greedy train_task=<TRAIN_ID>
```

Beam search:

```
python src/run_predict.py --multirun --config-name=GPT_beam task_name=ml1_GPT_beam train_task=<TRAIN_ID> generation_params.num_beams=2,3
```

Temperature sampling with multi-sequence aggregation:

```
python src/run_predict.py --config-name=GPT_temperature task_name=ml1_GPT_multisequence train_task=<TRAIN_ID> generation_params.temperature=0.3 generation_params.num_return_sequences=20
```

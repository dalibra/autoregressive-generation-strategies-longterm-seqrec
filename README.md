# Autoregressive Generation Strategies for Long-term Sequential Recommendations

This repository contains code for the paper ''**Autoregressive generation strategies for long-term sequential recommendations**''.

The goal of sequential recommendations is usually formulated in terms of next-item prediction. In this paper, we consider Transformer-based models for longer-term sequential recommendations. We introduce a generative approach to this task, which consists of training an autoregressive neural network on a classic next-item prediction task and autoregressive generation of recommendations item-by-item for the next N items. We explore different autoregressive generation strategies, including greedy decoding, beam search, and temperature sampling. In addition, we propose novel generation strategies based on multi-sequence generation with temperature sampling and subsequent aggregation. Experiments on diverse datasets with the GPT-2 model show that this approach can improve performance on longer time horizons.

## Usage

Install requirements:
```sh
pip install -r requirements.txt
```
This repository has been tested with `Python 3.9.16`. **Note, that the code doesn't work with `Python >= 3.10`, since `recommenders` library requires `Python < 3.10`**.

or you can directly download preprocessed datasets: [ML-20M](https://anonymfile.com/3rKP/ml-20m.csv), [Yelp](https://anonymfile.com/8Bzn/yelp.csv), [Steam](https://anonymfile.com/ry5Z/steam.csv), [Gowalla](https://anonymfile.com/4a1k/gowalla.csv), [Twitch-100k](https://anonymfile.com/mLX1/twitch.csv), [BeerAdvocate](https://anonymfile.com/k6RW/beer-advocate.csv).

We use [Hydra](https://hydra.cc/) for configuring the experements and [ClearML](`https://clear.ml/docs/latest/docs`) to log results.
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

BPR-MF code and experements are in a separate notebook `notebooks/BPR-MF.ipynb`.

Temperature sampling with multi-sequence aggregation:

```
python src/run_predict.py --config-name=GPT_temperature task_name=ml1_GPT_multisequence train_task=<TRAIN_ID> generation_params.temperature=0.3 generation_params.num_return_sequences=20
```

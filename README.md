# Autoregressive Generation Strategies for Long-term Sequential Recommendations

This repository contains code for the paper ''**Autoregressive generation strategies for long-term sequential recommendations**''.

The goal of sequential recommendations is usually formulated in terms of next-item prediction. In this paper, we consider Transformer-based models for longer-term sequential recommendations. We introduce a generative approach to this task, which consists of training an autoregressive neural network on a classic next-item prediction task and autoregressive generation of recommendations item-by-item for the next N items. We explore different autoregressive generation strategies, including greedy decoding, beam search, and temperature sampling. In addition, we propose novel generation strategies based on multi-sequence generation with temperature sampling and subsequent aggregation. Experiments on diverse datasets with the GPT-2 model show that this approach can improve performance on longer time horizons.

## Usage

Install requirements:
```sh
pip install -r requirements.txt
```
This repository has been tested with `Python 3.9.16`. **Note, that the code doesn't work with `Python >= 3.10` since the `recommenders` library requires `Python < 3.10`**.

Before running the code, create a `data/` folder and place the preprocessed datasets inside. You can see an example of the preprocessing in `notebooks/Example_preprocessing_ml-20m.ipynb`. You can also download preprocessed datasets directly: [ML-20M](https://anonymfile.com/3rKP/ml-20m.csv), [Yelp](https://anonymfile.com/8Bzn/yelp.csv), [Steam](https://anonymfile.com/ry5Z/steam.csv), [Gowalla](https://anonymfile.com/4a1k/gowalla.csv), [Twitch-100k](https://anonymfile.com/mLX1/twitch.csv), [BeerAdvocate](https://anonymfile.com/k6RW/beer-advocate.csv).

We use [Hydra](https://hydra.cc/) for configuring the experiments and [ClearML](`https://clear.ml/docs/latest/docs`) to log results.
All configurable parameters can be found in corresponding configs at `src/configs`, and also they can be overridden from the command line.

Below are examples of training GPT-2 and testing the obtained model with different strategies.AAAAAAa

Baselines:
```sh
# SASRec+
python src/run_train_predict.py --config-name=SASRec_train_predict data_path=data/ml-20m.csv task_name=ml-20m_SASRec_predict dataloader.test_batch_size=256
# BERT4Rec
python src/run_train_predict.py --config-name=BERT4Rec_train_predict data_path=data/ml-20m.csv task_name=ml-20m_BERT4Rec_predict dataloader.test_batch_size=256
```

BPR-MF code and experiments are in a separate notebook `notebooks/BPR-MF.ipynb`.


To optimize our process, we can train the model once and then deploy the trained version for subsequent tests, eliminating the need to retrain it with every run. For that, save task ID `<TRAIN_ID>` from the obtained `ClearML` training page.
```sh
# Train GPT-2 model
python src/run_train.py --config-name=GPT_train data_path=data/ml-20m.csv task_name=ml-20m_GPT_predict dataloader.test_batch_size=256
```
And then use `<TRAIN_ID>` as an argument for `train_task`:
```sh
# GPT Top-K prediction
python src/run_predict.py --config-name=GPT_predict task_name=ml-20m_GPT_predict train_task=<TRAIN_ID>
```


### Train Model

```
python src/run_train.py --config-name=GPT_train data_path=data/ml-1m.csv task_name=ml1_GPT_train seqrec_module.lr=1e-3
```

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

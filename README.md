# Autoregressive Generation Strategies for Top-K Sequential Recommendations

The goal of modern sequential recommender systems is often formulated in terms of next-item prediction. In this paper, we explore the applicability of generative transformer-based models for the Top-K sequential recommendation task, where the goal is to predict items a user is likely to interact with in the “near future”. 

We explore commonly used autoregressive generation strategies, including greedy decoding, beam search, and temperature sampling to evaluate their performance for Top-K sequential recommendation task. In addition, we propose novel Reciprocal Rank Aggregation (RRA) and Relevance Aggregation (RA) generation strategies based on multi-sequence generation with temperature sampling and subsequent aggregation. 

Experiments on diverse datasets give valuable insights regarding commonly used strategies' applicability and show that suggested approaches improve performance on longer time horizons compared to widely-used Top-K prediction approach and single-sequence autoregressive generation strategies. 

Reciprocal Rank Aggregation:
![RRA scheme](assets/strat1.png?raw=true)

Relevance Aggregation:
![RA scheme](assets/strat2.png?raw=true)

## Usage

Install requirements:
```sh
pip install -r requirements.txt
```
This repository has been tested with `Python 3.9.16`. **Note, that the current version of the code doesn't work with `Python >= 3.10` since the `recommenders` library requires `Python < 3.10`**.

Before running the code, create a `data/` folder and place the preprocessed datasets inside. You can see an example of the preprocessing in `notebooks/Example_preprocessing_ml-20m.ipynb`. You can also download preprocessed datasets directly: [ML-20M](https://disk.yandex.ru/d/bsp3rd-l_EpExA), [Yelp](https://disk.yandex.ru/d/UTKDilplnEV2iA), [Steam](https://disk.yandex.ru/d/4a2zDGsNnrR9rA), [Gowalla](https://disk.yandex.ru/d/K5K2CYuQF9KhMA), [Twitch-100k](https://disk.yandex.ru/d/lPiiN5ug0WQ3gw), [BeerAdvocate](https://disk.yandex.ru/d/8nImZhLxbLrkIw).

We use [Hydra](https://hydra.cc/) for configuring the experiments and [ClearML](`https://clear.ml/docs/latest/docs`) to log results.
All configurable parameters can be found in corresponding configs at `src/configs` and can be overridden from the command line.


## Examples

Below you can find examples of training the models and then testing them using different strategies mentioned in the paper.

#### Baselines:
```sh
# SASRec+
python src/run_train_predict.py --config-name=SASRec_train_predict data_path=data/ml-20m.csv task_name=ml-20m_SASRec_train_predict dataloader.test_batch_size=256 model_params.hidden_units=256

# BERT4Rec
python src/run_train_predict.py --config-name=BERT4Rec_train_predict data_path=data/ml-20m.csv task_name=ml-20m_BERT4Rec_train_predict dataloader.test_batch_size=256 model_params.hidden_size=256
```

BPR-MF code and experiments are in a separate notebook `notebooks/BPR_MF.ipynb`.

To optimize our process, we can train the model once and then deploy the trained version for subsequent tests, eliminating the need to retrain it with every run. For that, save task ID `<TRAIN_ID>` from the obtained `ClearML` training page.
```sh
# Train GPT-2 model
python src/run_train.py --config-name=GPT_train data_path=data/ml-20m.csv task_name=ml-20m_GPT_train dataloader.test_batch_size=256 model_params.n_embd=256
```
And then use `<TRAIN_ID>` as an argument for `train_task` parameter:
```sh
# GPT Top-K prediction
python src/run_predict.py --config-name=GPT_predict train_task=<TRAIN_ID> task_name=ml-20m_GPT_predict dataloader.test_batch_size=256
```

#### Generation strategies

```sh
# Greedy decoding
python src/run_predict.py --config-name=GPT_greedy train_task=<TRAIN_ID> task_name=ml-20m_GPT_greedy dataloader.test_batch_size=72

# Beam search with number of beams 2
python src/run_predict.py --config-name=GPT_beam train_task=<TRAIN_ID> task_name=ml-20m_GPT_beam generation_params.num_beams=2 dataloader.test_batch_size=72

# Temperature sampling with temperature 0.5
python src/run_predict.py --config-name=GPT_temperature train_task=<TRAIN_ID> task_name=ml-20m_GPT_temperature generation_params.temperature=0.5 dataloader.test_batch_size=72

# Reciprocal rank aggregation with 30 sequences and the best temperature with the best top_k
python src/run_predict.py --config-name=GPT_temperature train_task=<TRAIN_ID> task_name=ml-20m_GPT_multisequence generation_params.temperature=0.6 mode='reciprocal_rank_aggregation' generation_params.num_return_sequences=30 generation_params.top_k=20 dataloader.test_batch_size=72

# Relevance aggregation with 30 sequences and the best temperature 
python src/run_predict.py --config-name=GPT_temperature train_task=<TRAIN_ID> task_name=ml-20m_GPT_multisequence generation_params.temperature=0.7 mode='relevance_aggregation' generation_params.num_return_sequences=30 generation_params.top_k=0 dataloader.test_batch_size=72
```

#### Hyperparameters tuning

```sh
# Reciprocal rank aggregation with tuning top_k and temperature
python src/run_predict.py --config-name=GPT_RRA_Optuna train_task=<TRAIN_ID> task_name=ml-20m_GPT_multisequence dataloader.test_batch_size=72 --multirun 

# Relevance aggregation with tuning temperature
python src/run_predict.py --config-name=GPT_temperature train_task=<TRAIN_ID> task_name=ml-20m_GPT_multisequence generation_params.temperature='choice(1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0)' mode='relevance_aggregation' generation_params.num_return_sequences=30 generation_params.top_k=0 dataloader.test_batch_size=72 --multirun
```


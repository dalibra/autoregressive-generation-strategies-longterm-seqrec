"""
Run full experiment - train + predict.
"""

import time
import os

import hydra
import numpy as np
import pandas as pd

from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary, TQDMProgressBar)
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, BertConfig, BertModel

from datasets import CausalLMDataset, CausalLMPredictionDataset, PaddingCollateFn, MaskedLMDataset, MaskedLMPredictionDataset
from metrics import Evaluator
from modules import SeqRecHuggingface, SeqRec
from models import SASRec, BERT4Rec
from postprocess import preds2recs
from preprocess import add_time_idx


@hydra.main(version_base=None, config_path="configs", config_name="GPT_train_predict")
def main(config):

    print(OmegaConf.to_yaml(config))

    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if hasattr(config, 'project_name'):
        if hasattr(config, 'seed'):
            Task.set_random_seed(config.seed)
        else:
            Task.set_random_seed(None)
        task = Task.init(project_name=config.project_name, task_name=config.task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    train, validation, validation_full, test, item_count = prepare_data(config)
    train_loader, eval_loader = create_dataloaders(train, validation_full, config)
    model = create_model(config, item_count=item_count)
    start_time = time.time()
    trainer, seqrec_module = training(model, train_loader, eval_loader, config)
    training_time = time.time() - start_time
    print('training_time', training_time)

    if config.test_metrics:
        recs = predict(trainer, seqrec_module, train, config)
    else:
        recs = predict(trainer, seqrec_module,
                       train[train.user_id.isin(validation.user_id.unique())], config)
    
    if hasattr(config, 'optuna_metrics'):
        val_metrics = evaluate(recs, validation[validation.time_idx == 0], train, task, config, prefix='val')
        return val_metrics[val_metrics['metric_name'] == config.optuna_metrics]['metric_value'].values
    else:
        evaluate(recs, validation, train, task, config, prefix='val')
    if config.test_metrics:
        evaluate(recs, test, train, task, config, prefix='test')

    if task is not None:
        task.get_logger().report_single_value('training_time', training_time)
        task.upload_artifact('recs', recs)
        task.close()


def prepare_data(config):

    data = pd.read_csv(config.data_path)
    data = add_time_idx(data)

    # index 1 is used for masking value
    if config.model == 'BERT4Rec':
        data.item_id += 1

    train = data[data.time_idx_reversed >= config.last_n_items]
    test = data[data.time_idx_reversed < config.last_n_items]

    users_validation, users_test = train_test_split(
        test.user_id.unique(), test_size=0.5, random_state=42)
    validation = test[test.user_id.isin(users_validation)]
    test = test[test.user_id.isin(users_test)]

    train = add_time_idx(train)
    validation = add_time_idx(validation)
    test = add_time_idx(test)

    train2 = train[train.user_id.isin(users_validation)]
    validation2 = validation[validation.time_idx == 0]
    validation_full = pd.concat([train2, validation2])
    validation_full = add_time_idx(validation_full)

    item_count = data.item_id.max()

    return train, validation, validation_full, test, item_count


def create_dataloaders(train, validation, config):

    validation_size = config.dataloader.validation_size
    validation_users = validation.user_id.unique()
    if validation_size and (validation_size < len (validation_users)):
        
        np.random.seed(42)
        validation_users = np.random.choice(validation_users, size=validation_size, replace=False)
        validation = validation[validation.user_id.isin(validation_users)]
    
    train_dataset = MaskedLMDataset(train, **config['dataset']) if config.model == 'BERT4Rec' else CausalLMDataset(train, **config['dataset'])
    eval_dataset = MaskedLMPredictionDataset(validation, max_length=config.dataset.max_length, validation_mode=True) if config.model == 'BERT4Rec' else CausalLMPredictionDataset(validation, max_length=config.dataset.max_length, validation_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size,
                              shuffle=True, num_workers=config.dataloader.num_workers,
                              collate_fn=PaddingCollateFn())
    eval_loader = DataLoader(eval_dataset, batch_size=config.dataloader.test_batch_size,
                             shuffle=False, num_workers=config.dataloader.num_workers,
                             collate_fn=PaddingCollateFn())

    return train_loader, eval_loader


def create_model(config, item_count, weights_path=None):

    if config.model == 'GPT-2':
        gpt2_config = GPT2Config(vocab_size=item_count + 1, **config.model_params)
        model = GPT2LMHeadModel(gpt2_config)
    elif config.model == 'SASRec':
        model = SASRec(item_num=item_count, **config.model_params)
    elif config.model == 'BERT4Rec':
        model = BERT4Rec(vocab_size=item_count + 1, add_head=True, tie_weights=True, bert_config=config.model_params) #######################?
        
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    return model


def training(model, train_loader, eval_loader, config):

    if config.model == 'GPT-2':
        seqrec_module = SeqRecHuggingface(model, **config['seqrec_module'])
    elif config.model == 'SASRec':
        seqrec_module = SeqRec(model, **config['seqrec_module'])
    elif config.model == 'BERT4Rec':
        seqrec_module = SeqRec(model, **config['seqrec_module'])

    early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                   patience=config.patience, verbose=False)
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_ndcg",
                                 mode="max", save_weights_only=True)
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks=[early_stopping, model_summary, checkpoint, progress_bar]

    trainer = pl.Trainer(callbacks=callbacks, enable_checkpointing=True,
                         **config['trainer_params'])

    trainer.fit(model=seqrec_module,
            train_dataloaders=train_loader,
            val_dataloaders=eval_loader)

    seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])

    return trainer, seqrec_module


def predict(trainer, seqrec_module, data, config):

    if config.model == 'GPT-2':
        if config.generation:
            predict_dataset = CausalLMPredictionDataset(
                data, max_length=config.dataset.max_length - max(config.evaluator.top_k))
            
            predict_loader = DataLoader(
                predict_dataset, shuffle=False,
                collate_fn=PaddingCollateFn(left_padding=True),
                batch_size=config.dataloader.test_batch_size,
                num_workers=config.dataloader.num_workers)
            seqrec_module.set_predict_mode(generate=True, mode=config.mode, **config.generation_params)
        else:
            predict_dataset = CausalLMPredictionDataset(data, max_length=config.dataset.max_length)
            predict_loader = DataLoader(
                predict_dataset, shuffle=False,
                collate_fn=PaddingCollateFn(),
                batch_size=config.dataloader.test_batch_size,
                num_workers=config.dataloader.num_workers)
            seqrec_module.set_predict_mode(generate=False)
        
    elif config.model == 'SASRec':
        predict_dataset = CausalLMPredictionDataset(data, max_length=config.dataset.max_length)
        predict_loader = DataLoader(
            predict_dataset, shuffle=False,
            collate_fn=PaddingCollateFn(),
            batch_size=config.dataloader.test_batch_size,
            num_workers=config.dataloader.num_workers)
        
    elif config.model == 'BERT4Rec':
        predict_dataset = MaskedLMPredictionDataset(data, max_length=config.dataset.max_length)
        predict_loader = DataLoader(
            predict_dataset, shuffle=False,
            collate_fn=PaddingCollateFn(),
            batch_size=config.dataloader.test_batch_size,
            num_workers=config.dataloader.num_workers)

    seqrec_module.predict_top_k = max(config.evaluator.top_k)
    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)

    recs = preds2recs(preds)
    print('recs shape', recs.shape)

    return recs


def evaluate(recs, test, train, task, config, prefix='test'):

    evaluator = Evaluator(**config['evaluator'])

    metrics = evaluator.compute_metrics(test, recs, train)
    metrics = {prefix + '_' + key: value for key, value in metrics.items()}
    print(f'{prefix} metrics\n', metrics)

    compute_by_time_idx_flag = test['time_idx'].nunique() > 1
    if compute_by_time_idx_flag:
        metrics_by_time_idx = evaluator.compute_metrics_by_time_idx(test, recs)
        print(f'{prefix} metrics_by_time_idx\n', metrics_by_time_idx.to_string())
        metrics_by_time_idx_top_k_gt = evaluator.compute_metrics_by_time_idx(
            test, recs, top_k_gt=True)
        print(f'{prefix} metrics_by_time_idx_top_k_gt\n', metrics_by_time_idx_top_k_gt.to_string())

    if task:

        clearml_logger = task.get_logger()

        for key, value in metrics.items():
            clearml_logger.report_single_value(key, value)

        if compute_by_time_idx_flag:
            for metric_name in metrics_by_time_idx.columns:
                for i, value in metrics_by_time_idx[metric_name].to_dict().items():
                    clearml_logger.report_scalar(title=prefix + '_' + metric_name,
                                                 series='by_time_idx', value=value, iteration=i)
                for i, value in metrics_by_time_idx_top_k_gt[metric_name].to_dict().items():
                    clearml_logger.report_scalar(title=prefix + '_' + metric_name,
                                                 series='by_time_idx_top_k_gt',
                                                 value=value, iteration=i)

        metrics = pd.Series(metrics).to_frame().reset_index()
        metrics.columns = ['metric_name', 'metric_value']
        clearml_logger.report_table(title=f'{prefix}_metrics', series='dataframe',
                                    table_plot=metrics)
        task.upload_artifact(f'{prefix}_metrics', metrics)

        if compute_by_time_idx_flag:
            clearml_logger.report_table(title=f'{prefix}_metrics_by_time_idx', series='dataframe',
                                        table_plot=metrics_by_time_idx)
            task.upload_artifact(f'{prefix}_metrics_by_time_idx', metrics_by_time_idx)
            clearml_logger.report_table(title=f'{prefix}_metrics_by_time_idx_top_k_gt',
                                        series='dataframe',
                                        table_plot=metrics_by_time_idx_top_k_gt)
            task.upload_artifact(f'{prefix}_metrics_by_time_idx_top_k_gt',
                              metrics_by_time_idx_top_k_gt)
    return metrics


if __name__ == "__main__":

    main()

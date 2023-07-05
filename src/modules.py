"""
Pytorch Lightning Modules.
"""

from collections import Counter

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


class SeqRecBase(pl.LightningModule):

    def __init__(self, model, lr=1e-3, padding_idx=0,
                 predict_top_k=10, filter_seen=True):

        super().__init__()

        self.model = model
        self.lr = lr
        self.padding_idx = padding_idx
        self.predict_top_k = predict_top_k
        self.filter_seen = filter_seen

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)

        scores = scores.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        user_ids = batch['user_id'].detach().cpu().numpy()

        return {'preds': preds, 'scores': scores, 'user_ids': user_ids}

    def validation_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)
        metrics = self.compute_val_metrics(batch['target'], preds)

        self.log("val_ndcg", metrics['ndcg'], prog_bar=True)
        self.log("val_hit_rate", metrics['hit_rate'], prog_bar=True)
        self.log("val_mrr", metrics['mrr'], prog_bar=True)

    def make_prediction(self, batch):

        outputs = self.prediction_output(batch)

        input_ids = batch['input_ids']
        rows_ids = torch.arange(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        last_item_idx = (input_ids != self.padding_idx).sum(axis=1) - 1

        preds = outputs[rows_ids, last_item_idx, :]

        scores, preds = torch.sort(preds, descending=True)

        if self.filter_seen:
            seen_items = batch['full_history']
            preds, scores = self.filter_seen_items(preds, scores, seen_items)
        else:
            scores = scores[:, :self.predict_top_k]
            preds = preds[:, :self.predict_top_k]

        return preds, scores

    def filter_seen_items(self, preds, scores, seen_items):

        max_len = seen_items.size(1)
        scores = scores[:, :self.predict_top_k + max_len]
        preds = preds[:, :self.predict_top_k + max_len]

        final_preds, final_scores = [], []
        for i in range(preds.size(0)):
            not_seen_indexes = torch.isin(preds[i], seen_items[i], invert=True)
            pred = preds[i, not_seen_indexes][:self.predict_top_k]
            score = scores[i, not_seen_indexes][:self.predict_top_k]
            final_preds.append(pred)
            final_scores.append(score)

        final_preds = torch.vstack(final_preds)
        final_scores = torch.vstack(final_scores)

        return final_preds, final_scores

    def compute_val_metrics(self, targets, preds):

        ndcg, hit_rate, mrr = 0, 0, 0

        for i, pred in enumerate(preds):
            if torch.isin(targets[i], pred).item():
                hit_rate += 1
                rank = torch.where(pred == targets[i])[0].item() + 1
                ndcg += 1 / np.log2(rank + 1)
                mrr += 1 / rank

        hit_rate = hit_rate / len(targets)
        ndcg = ndcg / len(targets)
        mrr = mrr / len(targets)

        return {'ndcg': ndcg, 'hit_rate': hit_rate, 'mrr': mrr}


class SeqRecHuggingface(SeqRecBase):

    generate: bool = False
    generate_params: dict

    def training_step(self, batch, batch_idx):

        outputs = self.model(**batch)
        loss = outputs.loss

        return loss

    def prediction_output(self, batch):

        outputs = self.model(input_ids=batch['input_ids'],
                             attention_mask=batch['attention_mask'])

        return outputs.logits

    def set_predict_mode(self,
                         generate=False,
                         **generate_kwargs):
        """
        Set `predict` options.
        If `generate` is False, general predict method is used, which returns top-k most relevant next items.
        If `generate` is True, sequence is continued with `generate` method of HuggingFaceModel class.
        `generate_kwargs` are passed to model.generate(). Default generate_params are
        {"early_stopping": False, "no_repeat_ngram_size": 1}.
        `Generate` params are explained here: https://huggingface.co/blog/how-to-generate
        """
        self.generate = generate
        self.generate_params = {"early_stopping": False, "no_repeat_ngram_size": 1}
        if generate and generate_kwargs is not None:
            self.generate_params.update(generate_kwargs)

    def process_multiple_sequences(self, batch, preds, scores):
        """
        Combine multiple sequences generated for one user into one and leave top-k with maximal score.
        Score of an item is calculated as a sum of scores of an item in each sequence.
        """
        preds_batch, scores_batch = [], []
        num_seqs = self.generate_params["num_return_sequences"]
        for user_idx in range(batch['user_id'].shape[0]):
            dicts = [dict(zip(preds[user_idx * num_seqs + i, :].detach().cpu().numpy(),
                              scores[user_idx * num_seqs + i, :].detach().cpu().numpy())) for i in range(num_seqs)]
            combined_dict = dict(sum((Counter(d) for d in dicts), Counter()))
            preds_one, scores_one = list(
                zip(*sorted(combined_dict.items(), key=lambda x: x[1], reverse=True)[:preds.shape[1]]))
            preds_batch.append(preds_one)
            scores_batch.append(scores_one)
        return np.array(preds_batch), np.array(scores_batch)

    def predict_step(self, batch, batch_idx):
        user_ids = batch['user_id'].detach().cpu().numpy()

        if not self.generate or \
                ("num_return_sequences" not in self.generate_params
                 or self.generate_params["num_return_sequences"] == 1):
            if not self.generate:
                preds, scores = self.make_prediction(batch)
            else:
                preds, scores = self.make_prediction_generate(batch)
            scores = scores.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            return {'preds': preds, 'scores': scores, 'user_ids': user_ids}

        # `generate` with num_return_sequences > 1
        preds, scores = self.make_prediction_generate(batch)
        preds, scores = self.process_multiple_sequences(batch, preds, scores)
        return {'preds': preds, 'scores': scores, 'user_ids': user_ids}

    def make_prediction_generate(self, batch):
        """
        Continue sequence with `generate` method of HuggingFaceModel class.
        Batch should be left-padded e.g. with the PaddingCollateFn(left_padding=True).
        Input sequence may be cropped,
        maximum self.model.config.n_positions - self.predict_top_k last items are used as a sequence beginning.
        """
        seen_items = None
        if self.filter_seen:
            if batch['full_history'].shape[0] == 1:
                seen_items = batch['full_history'][
                    batch['full_history'] != self.padding_idx].detach().cpu().numpy().reshape(-1, 1).tolist()
            else:
                raise ValueError(
                    "Use batch_size=1 to continue sequence with `HuggingFaceModel.generate()` and `filter_seen` == True")

        seq = self.model.generate(
            batch['input_ids'][:, - self.model.config.n_positions + self.predict_top_k:].to(self.model.device),
            pad_token_id=self.padding_idx,
            max_new_tokens=self.predict_top_k,
            bad_words_ids=seen_items,
            **self.generate_params
            )
        preds = seq[:, - self.predict_top_k:]
        scores_one = torch.pow(torch.arange(self.predict_top_k, dtype=torch.long, device=self.model.device) + 1.,
                               -1).reshape(1, -1)
        scores = torch.tile(scores_one, [preds.shape[0], 1])
        return preds, scores


class SeqRec(SeqRecBase):

    def training_step(self, batch, batch_idx):

        outputs = self.model(batch['input_ids'], batch['attention_mask'])
        loss = self.compute_loss(outputs, batch)

        return loss

    def compute_loss(self, outputs, batch):

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))

        return loss

    def prediction_output(self, batch):

        return self.model(batch['input_ids'], batch['attention_mask'])


class SeqRecWithSampling(SeqRec):

    def __init__(self, model, lr=1e-3, loss='cross_entropy',
                 padding_idx=0, predict_top_k=10, filter_seen=True):

        super().__init__(model, lr, padding_idx, predict_top_k, filter_seen)

        self.loss = loss

        if hasattr(self.model, 'item_emb'):  # for SASRec
            self.embed_layer = self.model.item_emb
        elif hasattr(self.model, 'embed_layer'):  # for other models
            self.embed_layer = self.model.embed_layer


    def compute_loss(self, outputs, batch):

        # embed  and compute logits for negatives
        if batch['negatives'].ndim == 2:  # for full_negative_sampling=False
            # [N, M, D]
            embeds_negatives = self.embed_layer(batch['negatives'].to(torch.int32))
            # [N, T, D] * [N, D, M] -> [N, T, M]
            logits_negatives = torch.matmul(outputs, embeds_negatives.transpose(1, 2))
        elif batch['negatives'].ndim == 3:  # for full_negative_sampling=True
            # [N, T, M, D]
            embeds_negatives = self.embed_layer(batch['negatives'].to(torch.int32))
            # [N, T, 1, D] * [N, T, D, M] -> [N, T, 1, M] -> -> [N, T, M]
            logits_negatives = torch.matmul(
                outputs.unsqueeze(2), embeds_negatives.transpose(2, 3)).squeeze()
            if logits_negatives.ndim == 2:
                logits_negatives = logits_negatives.unsqueeze(2)

        # embed  and compute logits for positives
        # [N, T]
        labels = batch['labels'].clone()
        labels[labels == -100] = self.padding_idx
        # [N, T, D]
        embeds_labels = self.embed_layer(labels)
        # [N, T, 1, D] * [N, T, D, 1] -> [N, T, 1, 1] -> [N, T]
        logits_labels = torch.matmul(outputs.unsqueeze(2), embeds_labels.unsqueeze(3)).squeeze()

        # concat positives and negatives
        # [N, T, M + 1]
        logits = torch.cat([logits_labels.unsqueeze(2), logits_negatives], dim=-1)

        # prepare targets for loss
        if self.loss == 'cross_entropy':
            # [N, T]
            targets = batch['labels'].clone()
            targets[targets != -100] = 0
        elif self.loss == 'bce':
            # [N, T, M + 1]
            targets = torch.zeros_like(logits)
            targets[:, :, 0] = 1

        if self.loss == 'cross_entropy':
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
        elif self.loss == 'bce':
            # loss_fct = nn.BCEWithLogitsLoss()
            # loss = loss_fct(logits, targets)
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_fct(logits, targets)
            loss = loss[batch['labels'] != -100]
            loss = loss.mean()

        return loss

    def prediction_output(self, batch):

        outputs = self.model(batch['input_ids'], batch['attention_mask'])
        outputs = torch.matmul(outputs, self.embed_layer.weight.T)

        return outputs

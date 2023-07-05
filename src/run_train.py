"""
Train model.
"""

import time
import os

import hydra
import numpy as np
import pandas as pd
import torch
from clearml import Task
from omegaconf import OmegaConf
from run_train_predict import prepare_data, create_dataloaders, create_model, training



@hydra.main(version_base=None, config_path="configs", config_name="GPT_train")
def main(config):

    print(OmegaConf.to_yaml(config))

    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if hasattr(config, 'project_name'):
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

    if task is not None:
        task.get_logger().report_single_value('training_time', training_time)
        torch.save(seqrec_module.model.state_dict(), 'model.pt')
        task.upload_artifact('model', 'model.pt')
        task.close()


if __name__ == "__main__":

    main()

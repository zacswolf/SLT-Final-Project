#### IMPORT PACKAGES
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from utils.data_loader import data_provider
import matplotlib.pyplot as plt
from models.bi_lstm import bi_LSTM
from models.transformer import Transformer
from models.bert_inspired import BertInspired
from models.ast_models import ASTModel
from utils.tools import dotdict
from utils.data_loader import DataModule
import pytorch_lightning as pl
from exp.exp_main import ExpMain
import os


#### MODEL
model_type = "ssast"  # "bert_inspired" "transformer" or "biLSTM"

model = None
model_config = None
if model_type == "biLSTM":
    # Create model
    model_config = dotdict(
        {
            "input_dim": 128,
            "hidden_dim": 128,
            "output_dim": 9,
            "num_layers": 2,
            "model_type": model_type,
        }
    )
    # create model
    model = bi_LSTM(model_config)
    # model.to(device)
elif model_type == "transformer":
    model_config = dotdict(
        {
            "enc_in": 128,
            "dec_in": 128,
            "c_out": 9,
            "d_model": 128,
            "dropout": 0.05,
            "output_attention": False,
            "n_heads": 8,
            "d_ff": None,
            "activation": "gelu",
            "e_layers": 2,
            "d_layers": 1,
            "model_type": model_type,
        }
    )
    model = Transformer(model_config)
    # model.to(device)
elif model_type == "bert_inspired":
    model_config = dotdict(
        {
            "enc_in": (32, 16),  # (#windows, # mel filters)
            "c_out": 9,
            "d_model": 512,
            "dropout": 0.05,
            "output_attention": False,
            "n_heads": 8,
            "d_ff": None,
            "activation": "gelu",
            "e_layers": 12,
            "model_type": model_type,
        }
    )
    model = BertInspired(model_config)
    # model.to(device)
elif model_type == "ssast":
    model_config = dotdict(
        {
            # "enc_in": (32, 16), # (#windows, # mel filters)
            # "c_out": 9,
            # "d_model": 512,
            # "dropout": .05,
            # "output_attention": False,
            # "n_heads": 8,
            # "d_ff": None,
            # "activation": "gelu",
            # "e_layers": 12,
            # "model_type": model_type
            "label_dim": 9,
            "fshape": 16,
            "tshape": 16,
            "fstride": 10,
            "tstride": 10,
            "input_fdim": 128,  # we need data sets with 128 mel filters
            "input_tdim": 16,  # This can change depending on data set
            "model_size": "base",
        }
    )

    # label_dim=527,
    # fshape=128, tshape=2, fstride=128, tstride=2,
    # input_fdim=128, input_tdim=1024, model_size='base',
    # pretrain_stage=True, load_pretrained_mdl_path=None
    model = ASTModel(
        **model_config,
        pretrain_stage=False,
        load_pretrained_mdl_path="pretrained_mdls/SSAST-Base-Patch-400.pth"
    )


assert model is not None, "Didn't select a valid model"


#### DATA MODULE AND EXPIRIMENT
exp_config = dotdict(
    {
        "seq_len": 1,
        "pred_len": 1,
        "data_id": "16x32",
        "batch_size": 512,
        "learning_rate": 0.000001,
        "max_epochs": 5,
        "loss": "BCE",
        "num_mel": 128,
        "num_frames": 16,
        "segment_duration": 100,
    }
)
assert model_type != "ssast" or exp_config.seq_len == 1, "SSAST needs seq_len == 1"
assert exp_config.pred_len <= exp_config.seq_len

strategy = "dp"  # ["ddp", "ddp_spawn", "ddp_notebook", "ddp_fork", None]
# num_workers = 4  # os.cpu_count() * (strategy != "ddp_spawn")


pl.seed_everything(seed=123, workers=True)
data_module = DataModule(exp_config, num_workers=0)


# Intantiate Lightning Model
exp = ExpMain(model, exp_config)


#### CREATE TRAINER

strategy = "dp"
trainer_params = {
    # "auto_scale_batch_size": "power",
    # "auto_lr_find": True,
    "max_epochs": exp_config.max_epochs,
    "logger": True,
    "accelerator": "gpu",
    "devices": -1,
    "auto_select_gpus": True,
    "strategy": strategy,  # GPUS
}
trainer = pl.Trainer(**trainer_params)


# Tune model (noop unless auto_scale_batch_size or auto_lr_find)
tuner_result = trainer.tune(exp, datamodule=data_module)
if "lr_find" in tuner_result:
    tuner_result["lr_find"].plot(suggest=True)
if "scale_batch_size" in tuner_result:
    print("scale_batch_size:", tuner_result["scale_batch_size"])


trainer.logger.log_hyperparams(model_config | exp_config)

# Train Model
trainer.fit(exp, data_module)

# Test Model
trainer.test(exp, data_module)

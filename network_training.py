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
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
)
from utils.callbacks import CustomWriter
from pytorch_lightning.loggers import TensorBoardLogger
from exp.exp_main import ExpMain
import os

model_type = "bert_inspired"  # "bert_inspired" "transformer" or "biLSTM"

#### DATA MODULE AND EXPIRIMENT
exp_config = dotdict(
    {
        "seq_len": 1,
        "pred_len": 1,
        "batch_size": 512,
        "learning_rate": 0.00001,
        "max_epochs": 10,
        "loss": "BCE",
        "num_mel": 64,
        "num_frames": 32,
        "segment_duration": 100,
    }
)
assert model_type != "ssast" or exp_config.seq_len == 1, "SSAST needs seq_len == 1"
assert exp_config.pred_len <= exp_config.seq_len

strategy = "dp"  # ["ddp", "ddp_spawn", "ddp_notebook", "ddp_fork", None]
# num_workers = 4  # os.cpu_count() * (strategy != "ddp_spawn")


pl.seed_everything(seed=123, workers=True)
data_module = DataModule(exp_config, num_workers=4)


#### MODEL

model = None
model_config = None
if model_type == "biLSTM":
    # Create model
    model_config = dotdict(
        {
            "input_dim": 128,
            "hidden_dim": 128,
            "output_dim": 9,
            "num_layers": 3,
            "model_type": model_type,
        }
    )
    # create model
    model = bi_LSTM(model_config)
    # model.to(device)
elif model_type == "bert_inspired":
    model_config = dotdict(
        {
            "enc_in": (
                exp_config.num_mel,
                exp_config.num_frames,
            ),  # (#windows, # mel filters)
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
            "input_tdim": exp_config.num_freq,  # This can change depending on data set
            "model_size": "base",
        }
    )
    assert exp_config.num_mel == 128, "SSAST needs num_mel==128"

    model = ASTModel(
        **model_config,
        pretrain_stage=False,
        load_pretrained_mdl_path="pretrained_mdls/SSAST-Base-Patch-400.pth",
    )


assert model is not None, "Didn't select a valid model"


# Intantiate Lightning Model
exp = ExpMain(model, exp_config)


#### CREATE TRAINER

strategy = "dp"

# Define Callbacks
callbacks = []

# Checkpoint model with lowest val lost into checkpoint.ckpt
# Additionally, checkpoint final model into last.ckpt if args.no_early_stop
callbacks.append(
    ModelCheckpoint(
        filename="checkpoint",
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
)

# Print model details
callbacks.append(ModelSummary(max_depth=3))


callbacks.append(CustomWriter("result", "epoch"))

# Print all callbacks
print(
    "Callbacks:",
    list(map(lambda x: str(type(x))[str(type(x)).rfind(".") + 1 : -2], callbacks)),
)

setting = f"{model_type}_{exp_config.num_mel}_{exp_config.num_frames}_{exp_config.segment_duration}ms"
# Logger
logger = TensorBoardLogger(
    "lightning_logs",
    name=setting,
    flush_secs=30,
)

# Define Trainer Params
trainer_params = {
    # "auto_scale_batch_size": "power",
    # "auto_lr_find": True,
    # "fast_dev_run": True,  # For debugging
    # "profiler": "simple",  # For looking for bottlenecks
    # "detect_anomaly": True,
    "max_epochs": exp_config.max_epochs,
    "accelerator": "gpu",
    "devices": -1,
    "auto_select_gpus": True,
    "strategy": strategy,  # Multi GPU
    "default_root_dir": f"lightning_logs/{setting}",
    "enable_model_summary": False,
    "callbacks": callbacks,
    "logger": logger,
}
trainer = pl.Trainer(**trainer_params)

# Tune model (noop unless auto_scale_batch_size or auto_lr_find)
tuner_result = trainer.tune(exp, datamodule=data_module)
if "lr_find" in tuner_result:
    tuner_result["lr_find"].plot(suggest=True)
    plt.show()
if "scale_batch_size" in tuner_result:
    print("scale_batch_size:", tuner_result["scale_batch_size"])


trainer.logger.log_hyperparams(model_config | exp_config)

# Train Model
trainer.fit(exp, data_module)

# Test Model
trainer.test(exp, data_module)

# Predict and Save Results
results = trainer.predict(exp, data_module)

print("DONE!!!! Logged in:", trainer.log_dir)

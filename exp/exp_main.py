import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np


class ExpMain(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = self._select_criterion()

        self.learning_rate = config.learning_rate

    def _select_criterion(self):
        if self.config.loss == "F1":
            raise NotImplementedError()
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Only predict middle `pred_len` elements of `seq_len`
        num_rid = self.config.seq_len - self.config.pred_len
        l = num_rid // 2
        r = num_rid - l
        indicies = np.arange(l, self.config.seq_len - r)

        def middle_loss_closure(output, label):
            # Slice on the 2nd to last axis
            output_middle = output[..., indicies, :]
            label_middle = label[..., indicies, :]
            return criterion(output_middle, label_middle)

        return middle_loss_closure

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        feats, labels = batch
        feats = feats  # .float()
        labels = labels  # .float()

        output = self.model(feats)

        loss = self.criterion(output, labels)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop. It is independent of forward
        feats, labels = batch
        feats = feats.float()
        labels = labels.float()

        output = self.model(feats)

        loss = self.criterion(output, labels)
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        # validation_step defines the validation loop. It is independent of forward
        feats, labels = batch
        feats = feats.float()
        labels = labels.float()

        output = self.model(feats)

        loss = self.criterion(output, labels)
        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

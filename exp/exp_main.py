import torch
import torch.nn as nn
import pytorch_lightning as pl


class ExpMain(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.config = config

        self.learning_rate = config.learning_rate

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

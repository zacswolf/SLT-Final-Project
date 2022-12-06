import os
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
import numpy as np


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        folder_path = os.path.join(trainer.log_dir, "results/")
        os.makedirs(folder_path)

        data = {}
        for dataloader_idx, data_dl in enumerate(predictions):
            dataloader_data = {}
            dataset = trainer.predict_dataloaders[dataloader_idx].dataset

            for key in ["output", "labels"]:
                dataloader_data[key] = torch.concat(
                    [data_dl[i][key] for i in range(len(data_dl))]
                )

            for key in dataloader_data:
                np.save(
                    os.path.join(
                        folder_path, f"{key}_{dataset.flag}_{trainer.global_rank}.npy"
                    ),
                    dataloader_data[key],
                )

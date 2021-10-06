from pytorch_lightning import loggers
import torch
import wandb
import pandas as pd
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from data import DataModule
from model import ColaModel


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    cola_data = DataModule(cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length)
    cola_model = ColaModel(cfg.model.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./model",
        filename="best-checkpoint.ckpt",
        monitor="valid/loss",
        mode="min",
    )
    early_stopping_callcack = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics", entity="chesed")
    trainer = pl.Trainer(
        default_root_dir="logs",
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            SamplesVisualisationLogger(cola_data)            
        ],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.detrministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches = cfg.training.limit_val_batches,
    )

    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
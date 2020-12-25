import pytorch_lightning as pl

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor


def get_trainer(*, gpus, logger, max_epochs=0, **kwargs):
    callbacks, checkpoint_callback = None, None
    if isinstance(kwargs.get('resume_from_checkpoint'), str) \
       and max_epochs < 1:
        # disable logging for reloaded models with no training
        logger = None

    if logger is not None:
        callbacks = [LearningRateMonitor()]
        if max_epochs >= 1:
            # will inherit dirpath from logger
            checkpoint_callback = ModelCheckpoint()

    kwargs = {
        'track_grad_norm': 1,
        'val_check_interval': 1.0,
        **kwargs,
    }
    return pl.Trainer(
        gpus=gpus,
        weights_summary=None,
        terminate_on_nan=True,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
        max_epochs=max_epochs,
        **kwargs
    )

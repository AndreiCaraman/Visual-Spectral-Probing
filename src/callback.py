from transformers.integrations import WandbCallback
import wandb
import numpy as np
import matplotlib.pyplot as plt


class WandbCustomCallback(WandbCallback):
    """
    Custom WandbCallback to log filter weights plot
    """

    def __init__(self, trainer):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
        """
        super().__init__()
        self.trainer = trainer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def on_epoch_end(self, args, state, control, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)

        frq_filter = (
            [*self.trainer.model.named_parameters()][0][1].detach().cpu().numpy()
        )
        scaled_filter = self.sigmoid(frq_filter)

        fig, ax = plt.subplots()
        ax.bar(np.arange(frq_filter.shape[0]), scaled_filter)

        wandb.log({"Filter weights": fig})

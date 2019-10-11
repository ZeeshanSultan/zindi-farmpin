from tensorflow import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.loss_gap = []
        self.fig = plt.figure(figsize=(12,8))
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))

        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss: {:.3f} ({:.3f})".format(self.losses[-1],min(self.losses)))
        if "val_loss" in logs:
            self.loss_gap.append(logs.get("val_loss") - logs.get('loss'))
            self.val_losses.append(logs.get("val_loss"))

            plt.plot(
                self.x,
                self.val_losses,
                label="val_loss: {:.3f} ({:.3f})".format(self.val_losses[-1], min(self.val_losses)),
            )

            # Plot the gap between train and val loss
            plt.plot(
                self.x,
                self.loss_gap,
                label='loss gap: {:.3f}'.format(self.loss_gap[-1])
                )

            plt.axvline(x=self.val_losses.index(min(self.val_losses)))
        plt.legend()
        plt.show()


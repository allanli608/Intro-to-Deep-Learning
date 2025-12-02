import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def plot_history(history, title):
        """
        Plots training and validation loss and accuracy from a history dictionary.
        """
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Loss Plot
        ax[0].plot(epochs, history["train_loss"], label="Train Loss", marker=".")
        ax[0].plot(epochs, history["val_loss"], label="Val Loss", marker=".")
        ax[0].set_title(f"{title} - Loss")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].grid(True, linestyle="--", alpha=0.6)
        ax[0].legend()

        # Accuracy Plot
        ax[1].plot(epochs, history["train_acc"], label="Train Acc", marker=".")
        ax[1].plot(epochs, history["val_acc"], label="Val Acc", marker=".")
        ax[1].set_title(f"{title} - Accuracy")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].grid(True, linestyle="--", alpha=0.6)
        ax[1].legend()

        plt.tight_layout()
        plt.show()

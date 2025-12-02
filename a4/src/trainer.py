import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from prettytable import PrettyTable  # Use PrettyTable for tabular formatting
import os


class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def calculate_accuracy(self, preds, y):
        top_pred = preds.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        return correct.float() / y.shape[0]

    def train_epoch(self, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()

        for batch in iterator:
            optimizer.zero_grad()
            text, text_lengths = batch.text
            predictions = self.model(text, text_lengths)
            loss = criterion(predictions, batch.label)
            acc = self.calculate_accuracy(predictions, batch.label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()

        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text
                predictions = self.model(text, text_lengths)
                loss = criterion(predictions, batch.label)
                acc = self.calculate_accuracy(predictions, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def run_experiment(
        self,
        train_iter,
        val_iter,
        test_iter=None,
        epochs=10,
        lr=1e-3,
        name="Experiment",
        hyperparameters=None,
        save_weights=False,
    ):
        """
        Runs the experiment with training, validation, and optional testing.
        Allows passing hyperparameters (dict) for reporting.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(self.device)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_valid_loss = float("inf")
        total_start_time = time.time()

        print(f"Starting {name} | Params: {self.count_parameters():,}")

        # Store hyperparameters for reporting
        if hyperparameters is None:
            hyperparameters = {}
        # Ensure critical hyperparams are always stored
        hyperparameters["epochs"] = epochs
        hyperparameters["learning_rate"] = lr

        for epoch in range(
            epochs
        ):  # Use standard range if not using tqdm for epoch loop specifically, or wrap here
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_iter, optimizer, criterion)
            valid_loss, valid_acc = self.evaluate(val_iter, criterion)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f"{name}_best.pt")

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(valid_loss)
            history["val_acc"].append(valid_acc)

            # Print epoch summary
            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
            print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        self.plot_history(history, name)

        # Calculate final test metrics if test_iter is provided
        test_loss, test_acc = None, None
        if test_iter:
            # Load best model for testing
            self.model.load_state_dict(torch.load(f"{name}_best.pt"))
            test_loss, test_acc = self.evaluate(test_iter, criterion)

        if not save_weights:
            # Remove saved weights if not needed
            os.remove(f"{name}_best.pt")

        # Print final comprehensive summary table
        self.print_experiment_summary(
            name,
            total_time,
            self.count_parameters(),
            hyperparameters,
            history,
            test_loss,
            test_acc,
        )

        return history

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def plot_history(self, history, title):
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        ax[0].plot(epochs, history["train_loss"], label="Train Loss")
        ax[0].plot(epochs, history["val_loss"], label="Val Loss")
        ax[0].set_title(f"{title} - Loss")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # Accuracy
        ax[1].plot(epochs, history["train_acc"], label="Train Acc")
        ax[1].plot(epochs, history["val_acc"], label="Val Acc")
        ax[1].set_title(f"{title} - Accuracy")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        plt.show()

    def print_experiment_summary(
        self,
        name,
        total_time,
        num_params,
        hyperparameters,
        history,
        test_loss=None,
        test_acc=None,
    ):
        """
        Prints a tabular summary of the experiment using PrettyTable.
        """
        t = PrettyTable()
        t.field_names = ["Metric", "Value"]
        t.align = "l"  # Left align

        # Experiment Details
        t.add_row(["Experiment Name", name])
        t.add_row(["Total Training Time", f"{total_time:.2f}s"])
        t.add_row(["Trainable Parameters", f"{num_params:,}"])

        # Hyperparameters
        t.add_row(["--- Hyperparameters ---", ""])
        for key, value in hyperparameters.items():
            t.add_row([key, value])

        # Final Results (Last Epoch)
        t.add_row(["--- Final Results (Last Epoch) ---", ""])
        t.add_row(["Final Train Loss", f"{history['train_loss'][-1]:.4f}"])
        t.add_row(["Final Train Accuracy", f"{history['train_acc'][-1]*100:.2f}%"])
        t.add_row(["Final Val Loss", f"{history['val_loss'][-1]:.4f}"])
        t.add_row(["Final Val Accuracy", f"{history['val_acc'][-1]*100:.2f}%"])

        # Best Validation Results
        best_val_loss = min(history["val_loss"])
        best_val_acc = max(history["val_acc"])
        t.add_row(["Best Val Loss", f"{best_val_loss:.4f}"])
        t.add_row(["Best Val Accuracy", f"{best_val_acc*100:.2f}%"])

        # Test Results (Optional)
        if test_loss is not None:
            t.add_row(["--- Test Results (Best Model) ---", ""])
            t.add_row(["Test Loss", f"{test_loss:.4f}"])
            t.add_row(["Test Accuracy", f"{test_acc*100:.2f}%"])

        print(t)

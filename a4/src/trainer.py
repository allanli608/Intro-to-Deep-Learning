import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
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
        epochs=10,
        lr=1e-3,
        name="Experiment",
        hyperparameters=None,
        save_weights=True,
    ):
        """
        Runs the training loop.
        Returns:
            history: dict containing lists of loss/acc per epoch
            summary: dict containing scalar metrics for final reporting
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(self.device)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_valid_loss = float("inf")
        total_start_time = time.time()

        print(f"Starting {name} | Params: {self.count_parameters():,}")

        # Initialize summary dict with metadata
        if hyperparameters is None:
            hyperparameters = {}
        summary = {
            "Model Name": name,
            "Parameters": self.count_parameters(),
            **hyperparameters,
        }

        for epoch in range(epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_iter, optimizer, criterion)
            valid_loss, valid_acc = self.evaluate(val_iter, criterion)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if save_weights:
                    # Create directory if it doesn't exist
                    os.makedirs("./trained_weights", exist_ok=True)
                    torch.save(
                        self.model.state_dict(), f"./trained_weights/{name}_best.pt"
                    )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(valid_loss)
            history["val_acc"].append(valid_acc)

            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
            print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

        total_time = time.time() - total_start_time

        # Populate final summary metrics
        summary["Training Time (s)"] = round(total_time, 2)
        summary["Final Train Loss"] = round(history["train_loss"][-1], 4)
        summary["Final Train Acc"] = round(history["train_acc"][-1], 4)
        summary["Final Val Loss"] = round(history["val_loss"][-1], 4)
        summary["Final Val Acc"] = round(history["val_acc"][-1], 4)
        summary["Best Val Loss"] = round(min(history["val_loss"]), 4)
        summary["Best Val Acc"] = round(max(history["val_acc"]), 4)

        return history, summary

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

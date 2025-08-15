from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

class DataManager:
    def __init__(self):
        self.images_True = []
        self.images_False = []

    def load_images(self, image_dir):
        dirist = [x for x in os.listdir(image_dir) if not x.startswith(".")]
        for folder_name in dirist:
            folder_dir = os.path.join(image_dir, folder_name)
            for file_name in os.listdir(folder_dir):
                file_dir = os.path.join(folder_dir, file_name)
                label = 1 if file_dir.strip().endswith("True.png") else 0
                img = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
                if label == 1:
                    self.images_True.append(img)
                else:
                    self.images_False.append(img)
        self.images_True = np.array(self.images_True)  # NOTE: Apparently Kelly does not scale betwen 0 and 1
        self.images_False = np.array(self.images_False)
        print(f"TRUE: {len(self.images_True)} images with shape {self.images_True[0].shape}")
        print(f"FALSE: {len(self.images_False)} images with shape {self.images_False[0].shape}")

    @staticmethod
    def balance_pair(pos, neg, random_state=42):
        rng = np.random.default_rng(random_state)
        n_pos, n_neg = len(pos), len(neg)
        if n_pos == n_neg:
            return pos, neg
        if n_pos > n_neg:
            keep = rng.choice(n_pos, size=n_neg, replace=False)
            return pos[keep], neg
        else:
            keep = rng.choice(n_neg, size=n_pos, replace=False)
            return pos, neg[keep]

    def split_data(self):
        # images_True -> positives (label 1), images_False -> negatives (label 0)
        images_True_bal, images_False_bal = self.balance_pair(self.images_True,
                                                              self.images_False,
                                                              random_state=42)
        true_labels = np.ones(len(images_True_bal), dtype=np.int8)
        false_labels = np.zeros(len(images_False_bal), dtype=np.int8)

        # Stack + split
        images = np.concatenate([images_True_bal, images_False_bal], axis=0)
        labels = np.concatenate([true_labels, false_labels], axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print("Balanced counts:", dict(zip(*np.unique(labels, return_counts=True))))
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Image shape: {X_train[0].shape}")
        return X_train, X_test, y_train, y_test


class ModelPerformanceIndicators:
    @staticmethod
    def compute_error_matrix(test_predictions, y_test, cutoff=0.5):
        y_test = y_test.astype(int)
        binary_predictions = np.where(test_predictions > cutoff, 1, 0)
        # Compute confusion matrix
        cm = confusion_matrix(y_test, binary_predictions)

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix (Cutoff = {cutoff:.2f})")
        plt.show()
        return cm

    @staticmethod
    def compute_basic_metrics(test_predictions, y_test, cutoff=0.5, average="binary"):
        y_test = np.asarray(y_test)
        binary_predictions = np.where(test_predictions > cutoff, 1, 0)
        metrics = {
            "accuracy": accuracy_score(y_test, binary_predictions),
            "precision": precision_score(y_test, binary_predictions, average=average, zero_division=0),
            "recall": recall_score(y_test, binary_predictions, average=average, zero_division=0),
            "f1": f1_score(y_test, binary_predictions, average=average, zero_division=0),
        }
        return metrics

    @staticmethod
    def save_pdf(test_predictions, y_test, model_name):
        error_matrix = ModelPerformanceIndicators.compute_error_matrix(test_predictions, y_test)
        metrics = ModelPerformanceIndicators.compute_basic_metrics(test_predictions, y_test)
        metrics = {k: round(v, 5) for k, v in metrics.items()}
        output_dir = f"performance_reports/{model_name}_report.pdf"
        # Save to PDF
        with PdfPages(output_dir) as pdf:
            # --- Page 1: Confusion Matrix ---
            y_test_int = np.asarray(y_test).astype(int)
            binary_predictions = np.where(test_predictions > 0.5, 1, 0)
            disp = ConfusionMatrixDisplay(confusion_matrix=error_matrix, display_labels=[0, 1])
            disp.plot(cmap="Blues", values_format="d")
            plt.title("Confusion Matrix")
            pdf.savefig()  # Save the current figure
            plt.close()

            df_metrics = pd.DataFrame([metrics])
            fig, ax = plt.subplots()
            ax.axis("off")
            tbl = ax.table(
                cellText=df_metrics.values,
                colLabels=df_metrics.columns,
                loc="center",
                cellLoc="center"
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.2, 1.2)
            plt.title("Performance Metrics", pad=20)
            pdf.savefig()
            plt.close()
        print(f"✅ Model report saved as '{output_dir}'")
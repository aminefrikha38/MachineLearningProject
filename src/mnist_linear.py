import numpy as np
from tensorflow.keras.datasets import mnist
from src.utils import one_hot, softmax, cross_entropy, accuracy

import matplotlib.pyplot as plt
import os

def run_mnist_linear():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0

    Y_train = one_hot(y_train, 10)

    W = np.random.randn(784, 10) * 0.01
    b = np.zeros((1, 10))

    lr = 0.1
    epochs = 20

    loss_history = []
    acc_history = []
    os.makedirs("outputs/figures", exist_ok=True)

    for epoch in range(epochs):
        logits = X_train @ W + b
        probs = softmax(logits)

        loss = cross_entropy(Y_train, probs)

        dW = X_train.T @ (probs - Y_train) / X_train.shape[0]
        db = np.mean(probs - Y_train, axis=0, keepdims=True)

        W -= lr * dW
        b -= lr * db

        train_pred = np.argmax(probs, axis=1)
        train_acc = accuracy(y_train, train_pred)
        loss_history.append(loss)
        acc_history.append(train_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train accuracy: {train_acc:.4f}")

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Linear Model - Loss")
    plt.savefig("outputs/figures/linear_loss.png")
    plt.close()

    plt.figure()
    plt.plot(acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Linear Model - Training Accuracy")
    plt.savefig("outputs/figures/linear_accuracy.png")
    plt.close()

    test_logits = X_test @ W + b
    test_pred = np.argmax(test_logits, axis=1)
    test_acc = accuracy(y_test, test_pred)

    os.makedirs("outputs/figures", exist_ok=True)

    # Find wrong predictions
    wrong_indices = np.where(test_pred != y_test)[0][:9]

    plt.figure(figsize=(8, 8))

    for i, idx in enumerate(wrong_indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        plt.title(f"True: {y_test[idx]} | Pred: {test_pred[idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/figures/linear_misclassified.png")
    plt.close()

    print("\nFinal result:")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test error rate: {1 - test_acc:.4f}")
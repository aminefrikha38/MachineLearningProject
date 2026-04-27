import numpy as np
from tensorflow.keras.datasets import mnist
from src.utils import one_hot, softmax, cross_entropy, accuracy

import matplotlib.pyplot as plt
import os

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def run_mnist_nn():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0

    Y_train = one_hot(y_train, 10)

    input_size = 784
    hidden_size = 128
    output_size = 10

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
    b2 = np.zeros((1, output_size))

    lr = 0.1
    epochs = 10
    batch_size = 128
    n = X_train.shape[0]

    loss_history = []
    acc_history = []
    os.makedirs("outputs/figures", exist_ok=True)

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_train = X_train[indices]
        Y_train = Y_train[indices]
        y_train = y_train[indices]

        for start in range(0, n, batch_size):
            end = start + batch_size
            X_batch = X_train[start:end]
            Y_batch = Y_train[start:end]

            z1 = X_batch @ W1 + b1
            a1 = relu(z1)

            z2 = a1 @ W2 + b2
            probs = softmax(z2)

            dz2 = (probs - Y_batch) / X_batch.shape[0]
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ W2.T
            dz1 = da1 * relu_derivative(z1)

            dW1 = X_batch.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        z1_full = X_train @ W1 + b1
        a1_full = relu(z1_full)
        z2_full = a1_full @ W2 + b2
        probs_full = softmax(z2_full)

        loss = cross_entropy(Y_train, probs_full)
        train_pred = np.argmax(probs_full, axis=1)
        train_acc = accuracy(y_train, train_pred)

        loss_history.append(loss)
        acc_history.append(train_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train accuracy: {train_acc:.4f}")

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Neural Network - Loss")
    plt.savefig("outputs/figures/nn_loss.png")
    plt.close()

    plt.figure()
    plt.plot(acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Neural Network - Training Accuracy")
    plt.savefig("outputs/figures/nn_accuracy.png")
    plt.close()


    z1_test = X_test @ W1 + b1
    a1_test = relu(z1_test)
    z2_test = a1_test @ W2 + b2

    test_pred = np.argmax(z2_test, axis=1)
    test_acc = accuracy(y_test, test_pred)

    wrong_indices = np.where(test_pred != y_test)[0][:9]

    plt.figure(figsize=(8, 8))

    for i, idx in enumerate(wrong_indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        plt.title(f"True: {y_test[idx]} | Pred: {test_pred[idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/figures/nn_misclassified.png")
    plt.close()

    print("\nFinal result:")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test error rate: {1 - test_acc:.4f}")
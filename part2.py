import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS     = 30
LR         = 0.001
DATA_DIR   = "./data"

print(f"Using device: {DEVICE}")


def get_dataloaders():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training samples : {len(train_set)}")
    print(f"Test samples     : {len(test_set)}")
    return train_loader, test_loader


class CIFAR10_CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,  out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(8 * 8 * 64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}\n")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += inputs.size(0)

    return running_loss / total, 1 - correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += inputs.size(0)

    return running_loss / total, 1 - correct / total


def train(model, train_loader, test_loader, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    history = {"train_loss": [], "train_err": [], "test_loss": [], "test_err": []}

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Err':>10}"
          f"  {'Test Loss':>10}  {'Test Err':>10}")
    print("-" * 57)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_err = train_one_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_err = evaluate(model, test_loader, criterion)
        scheduler.step(te_loss)

        history["train_loss"].append(tr_loss)
        history["train_err"].append(tr_err)
        history["test_loss"].append(te_loss)
        history["test_err"].append(te_err)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_err:>10.4f}"
              f"  {te_loss:>10.4f}  {te_err:>10.4f}")

    return history


CLASSES = ["airplane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "boat", "truck"]


def plot_history(history, save_path="training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["test_loss"],  label="Test loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss curves"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, [e * 100 for e in history["train_err"]], label="Train error %")
    ax2.plot(epochs, [e * 100 for e in history["test_err"]],  label="Test error %")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Error rate (%)")
    ax2.set_title("Error rate curves"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved -> {save_path}")


def confusion_matrix_plot(model, loader, save_path="confusion_matrix.png"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            preds  = model(inputs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    n_classes = len(CLASSES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true, pred] += 1

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalised confusion matrix - CIFAR-10 CNN")

    thresh = cm_norm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved -> {save_path}")

    print("\nPer-class accuracy:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:>10s}: {cm_norm[i, i] * 100:.1f}%")

    return cm


def show_misclassified(model, test_loader, n=10, save_path="misclassified.png"):
    model.eval()
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    wrong_imgs, wrong_true, wrong_pred = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(DEVICE)).cpu()
            preds   = outputs.argmax(dim=1)
            mask    = preds != labels
            wrong_imgs.extend(inputs[mask])
            wrong_true.extend(labels[mask].tolist())
            wrong_pred.extend(preds[mask].tolist())
            if len(wrong_imgs) >= n:
                break

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i >= len(wrong_imgs):
            ax.axis("off"); continue
        img = wrong_imgs[i] * std + mean
        img = img.permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(img)
        ax.set_title(f"True: {CLASSES[wrong_true[i]]}\nPred: {CLASSES[wrong_pred[i]]}", fontsize=8)
        ax.axis("off")

    plt.suptitle("Sample of misclassified images", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Misclassified samples saved -> {save_path}")


def main():
    train_loader, test_loader = get_dataloaders()

    model = CIFAR10_CNN().to(DEVICE)
    print(model)
    count_parameters(model)

    history = train(model, train_loader, test_loader, epochs=EPOCHS)

    final_train_err = history["train_err"][-1] * 100
    final_test_err  = history["test_err"][-1]  * 100
    print(f"\n{'='*40}")
    print(f"Final train error rate : {final_train_err:.2f}%")
    print(f"Final test  error rate : {final_test_err:.2f}%")
    print(f"Final test  accuracy   : {100 - final_test_err:.2f}%")
    print(f"{'='*40}\n")

    torch.save(model.state_dict(), "cifar10_cnn.pth")
    print("Model weights saved -> cifar10_cnn.pth")

    plot_history(history)
    confusion_matrix_plot(model, test_loader)
    show_misclassified(model, test_loader)

    print("\n-- Comparison with published results --")
    print(f"{'Model':<52} {'Error (%)':>10}")
    print("-" * 64)
    ref = [
        ("Convolutional Deep Belief Networks (2010)", 21.10),
        ("Maxout Networks (2013)",                     9.38),
        ("Fractional Max-Pooling (2014)",               3.47),
        ("Densely Connected CNN (2016)",                3.46),
        ("Coupled Ensembles (2017)",                    2.68),
        ("Vision Transformer 16x16 (2021)",             0.50),
        ("Our CNN (this project)",          final_test_err),
    ]
    for name, err in ref:
        marker = " <" if "Our" in name else ""
        print(f"  {name:<50} {err:>8.2f}%{marker}")


if __name__ == "__main__":
    main()
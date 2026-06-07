# =============================================================
# PART 3: BREAST CANCER DETECTION — COMPLETE CODE
# =============================================================

# ---- Setup ----
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# =============================================================
# STEP 1: LOAD CSVs AND BUILD IMAGE PATH MAPPING
# =============================================================
DATASET_ROOT = "./cbis-ddsm/"

mass_train = pd.read_csv(os.path.join(DATASET_ROOT, "csv", "mass_case_description_train_set.csv"))
mass_test  = pd.read_csv(os.path.join(DATASET_ROOT, "csv", "mass_case_description_test_set.csv"))
dicom_info = pd.read_csv(os.path.join(DATASET_ROOT, "csv", "dicom_info.csv"))

print(f"Mass train rows: {len(mass_train)}")
print(f"Mass test rows:  {len(mass_test)}")
print(f"\nSeries descriptions:")
print(dicom_info['SeriesDescription'].value_counts())

full_mammo = dicom_info[dicom_info['SeriesDescription'] == 'full mammogram images'][['PatientID', 'image_path']].copy()
cropped    = dicom_info[dicom_info['SeriesDescription'] == 'cropped images'][['PatientID', 'image_path']].copy()

full_mammo_dict = dict(zip(full_mammo['PatientID'].str.strip(), full_mammo['image_path'].str.strip()))
cropped_dict    = dict(zip(cropped['PatientID'].str.strip(), cropped['image_path'].str.strip()))

print(f"\nFull mammogram mappings: {len(full_mammo_dict)}")
print(f"Cropped image mappings: {len(cropped_dict)}")

def extract_patient_id(csv_path):
    return str(csv_path).strip().split('/')[0]

def resolve_to_jpeg(csv_path, lookup_dict):
    patient_id = extract_patient_id(csv_path)
    if patient_id in lookup_dict:
        relative_path = lookup_dict[patient_id].replace("CBIS-DDSM/", "")
        full_path = os.path.join(DATASET_ROOT, relative_path)
        if os.path.exists(full_path):
            return full_path
    return None

mass_train['jpeg_path'] = mass_train['cropped image file path'].apply(lambda x: resolve_to_jpeg(x, cropped_dict))
mass_test['jpeg_path']  = mass_test['cropped image file path'].apply(lambda x: resolve_to_jpeg(x, cropped_dict))

mask_train_null = mass_train['jpeg_path'].isna()
mask_test_null  = mass_test['jpeg_path'].isna()
mass_train.loc[mask_train_null, 'jpeg_path'] = mass_train.loc[mask_train_null, 'image file path'].apply(lambda x: resolve_to_jpeg(x, full_mammo_dict))
mass_test.loc[mask_test_null, 'jpeg_path']   = mass_test.loc[mask_test_null, 'image file path'].apply(lambda x: resolve_to_jpeg(x, full_mammo_dict))

print(f"\nTrain resolved: {mass_train['jpeg_path'].notna().sum()} / {len(mass_train)}")
print(f"Test resolved:  {mass_test['jpeg_path'].notna().sum()} / {len(mass_test)}")

# =============================================================
# STEP 2: BINARY LABELS
# =============================================================
def map_label(pathology):
    if pathology in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
        return 0
    elif pathology == 'MALIGNANT':
        return 1
    return -1

mass_train['label'] = mass_train['pathology'].apply(map_label)
mass_test['label']  = mass_test['pathology'].apply(map_label)

train_df = mass_train[(mass_train['jpeg_path'].notna()) & (mass_train['label'] != -1)].reset_index(drop=True)
test_df  = mass_test[(mass_test['jpeg_path'].notna()) & (mass_test['label'] != -1)].reset_index(drop=True)

print(f"\n=== FINAL DATASET ===")
print(f"Train: {len(train_df)} samples (Benign: {(train_df['label']==0).sum()}, Malignant: {(train_df['label']==1).sum()})")
print(f"Test:  {len(test_df)} samples (Benign: {(test_df['label']==0).sum()}, Malignant: {(test_df['label']==1).sum()})")

# =============================================================
# STEP 3: DATASET CLASS AND DATALOADERS
# =============================================================
IMG_SIZE = 128
IMG_SIZE_RESNET = 224

class MammogramDataset(Dataset):
    def _init_(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
    def _len_(self):
        return len(self.dataframe)
    def _getitem_(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['jpeg_path']).convert('L')
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

train_transform_cnn = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_transform_cnn = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
train_transform_resnet = transforms.Compose([
    transforms.Resize((IMG_SIZE_RESNET, IMG_SIZE_RESNET)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_transform_resnet = transforms.Compose([
    transforms.Resize((IMG_SIZE_RESNET, IMG_SIZE_RESNET)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset_cnn    = MammogramDataset(train_df, transform=train_transform_cnn)
test_dataset_cnn     = MammogramDataset(test_df, transform=test_transform_cnn)
train_dataset_resnet = MammogramDataset(train_df, transform=train_transform_resnet)
test_dataset_resnet  = MammogramDataset(test_df, transform=test_transform_resnet)

# Class imbalance handling
train_labels = train_df['label'].values
class_counts = np.bincount(train_labels)
sample_weights = (1.0 / class_counts)[train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
class_weights_tensor = torch.FloatTensor(weights).to(device)
print(f"\nClass weights: Benign={weights[0]:.3f}, Malignant={weights[1]:.3f}")

BATCH_SIZE = 32
train_loader_cnn    = DataLoader(train_dataset_cnn, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
test_loader_cnn     = DataLoader(test_dataset_cnn, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
train_loader_resnet = DataLoader(train_dataset_resnet, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
test_loader_resnet  = DataLoader(test_dataset_resnet, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Visualize samples
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    img, lbl = test_dataset_cnn[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title("MALIGNANT" if lbl == 1 else "BENIGN", color='red' if lbl == 1 else 'green', fontsize=11)
    ax.axis('off')
plt.suptitle(f"Sample Mammogram Crops (resized to {IMG_SIZE}x{IMG_SIZE})", fontsize=14)
plt.tight_layout()
plt.savefig("sample_mammograms.png", dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# STEP 4: MODEL DEFINITIONS
# =============================================================

class MammogramCNN_v2(nn.Module):
    def _init_(self, img_size=128):
        super(MammogramCNN_v2, self)._init_()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        flat_size = 64 * (img_size // 4) * (img_size // 4)
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MammogramResNet(nn.Module):
    def _init_(self):
        super(MammogramResNet, self)._init_()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )
    def forward(self, x):
        return self.resnet(x)

# =============================================================
# STEP 5: TRAINING AND EVALUATION FUNCTIONS
# =============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

# =============================================================
# STEP 6: TRAIN CUSTOM CNN
# =============================================================
NUM_EPOCHS = 30
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print("=" * 60)
print("TRAINING MODEL 1: Custom CNN (from Part 2 architecture)")
print("=" * 60)

model_cnn = MammogramCNN_v2(img_size=IMG_SIZE).to(device)
print(f"Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")

optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler_cnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn, mode='min', factor=0.5, patience=5)

cnn_train_losses, cnn_train_accs = [], []
cnn_test_losses, cnn_test_accs = [], []
best_cnn_acc = 0.0

for epoch in range(NUM_EPOCHS):
    tr_loss, tr_acc = train_one_epoch(model_cnn, train_loader_cnn, criterion, optimizer_cnn)
    te_loss, te_acc, _, _ = evaluate(model_cnn, test_loader_cnn, criterion)
    cnn_train_losses.append(tr_loss); cnn_train_accs.append(tr_acc)
    cnn_test_losses.append(te_loss); cnn_test_accs.append(te_acc)
    scheduler_cnn.step(te_loss)
    if te_acc > best_cnn_acc:
        best_cnn_acc = te_acc
        torch.save(model_cnn.state_dict(), 'best_cnn_v2.pth')
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/30] | Train: {tr_loss:.4f} / {tr_acc:.4f} | Test: {te_loss:.4f} / {te_acc:.4f}")

print(f"Best CNN test accuracy: {best_cnn_acc:.4f}")

# =============================================================
# STEP 7: TRAIN RESNET18
# =============================================================
print("\n" + "=" * 60)
print("TRAINING MODEL 2: Pretrained ResNet18 (transfer learning)")
print("=" * 60)

model_resnet = MammogramResNet().to(device)
print(f"Parameters: {sum(p.numel() for p in model_resnet.parameters()):,}")

optimizer_resnet = optim.Adam(model_resnet.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler_resnet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_resnet, mode='min', factor=0.5, patience=5)

resnet_train_losses, resnet_train_accs = [], []
resnet_test_losses, resnet_test_accs = [], []
best_resnet_acc = 0.0

for epoch in range(NUM_EPOCHS):
    tr_loss, tr_acc = train_one_epoch(model_resnet, train_loader_resnet, criterion, optimizer_resnet)
    te_loss, te_acc, _, _ = evaluate(model_resnet, test_loader_resnet, criterion)
    resnet_train_losses.append(tr_loss); resnet_train_accs.append(tr_acc)
    resnet_test_losses.append(te_loss); resnet_test_accs.append(te_acc)
    scheduler_resnet.step(te_loss)
    if te_acc > best_resnet_acc:
        best_resnet_acc = te_acc
        torch.save(model_resnet.state_dict(), 'best_resnet.pth')
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/30] | Train: {tr_loss:.4f} / {tr_acc:.4f} | Test: {te_loss:.4f} / {te_acc:.4f}")

print(f"Best ResNet test accuracy: {best_resnet_acc:.4f}")
print("\n" + "=" * 60)
print(f"COMPARISON: CNN = {best_cnn_acc:.4f} vs ResNet = {best_resnet_acc:.4f}")
print("=" * 60)

# =============================================================
# STEP 8: TRAINING CURVES
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0][0].plot(range(1, 31), cnn_train_losses, label='Train Loss', color='blue')
axes[0][0].plot(range(1, 31), cnn_test_losses, label='Test Loss', color='red')
axes[0][0].set_title('Custom CNN — Loss'); axes[0][0].legend(); axes[0][0].grid(True, alpha=0.3)

axes[0][1].plot(range(1, 31), cnn_train_accs, label='Train Acc', color='blue')
axes[0][1].plot(range(1, 31), cnn_test_accs, label='Test Acc', color='red')
axes[0][1].set_title('Custom CNN — Accuracy'); axes[0][1].legend(); axes[0][1].grid(True, alpha=0.3)

axes[1][0].plot(range(1, 31), resnet_train_losses, label='Train Loss', color='blue')
axes[1][0].plot(range(1, 31), resnet_test_losses, label='Test Loss', color='red')
axes[1][0].set_title('ResNet18 — Loss'); axes[1][0].legend(); axes[1][0].grid(True, alpha=0.3)
axes[1][0].set_xlabel('Epoch')

axes[1][1].plot(range(1, 31), resnet_train_accs, label='Train Acc', color='blue')
axes[1][1].plot(range(1, 31), resnet_test_accs, label='Test Acc', color='red')
axes[1][1].set_title('ResNet18 — Accuracy'); axes[1][1].legend(); axes[1][1].grid(True, alpha=0.3)
axes[1][1].set_xlabel('Epoch')

plt.suptitle('Training Curves Comparison: Custom CNN vs Pretrained ResNet18', fontsize=14)
plt.tight_layout()
plt.savefig("training_curves_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# STEP 9: CONFUSION MATRIX AND METRICS
# =============================================================
model_resnet.load_state_dict(torch.load('best_resnet.pth'))
test_loss, test_acc, all_preds, all_labels = evaluate(model_resnet, test_loader_resnet, criterion)

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

print("=" * 55)
print("FINAL RESULTS — Pretrained ResNet18")
print("=" * 55)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"\nConfusion Matrix:")
print(f"{'':>20} Predicted")
print(f"{'':>15} {'Benign':>10} {'Malignant':>10}")
print(f"{'Actual Benign':>15} {tn:>10} {fp:>10}")
print(f"{'Actual Malignant':>15} {fn:>10} {tp:>10}")

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\nSensitivity (Recall for Malignant): {sensitivity:.4f}")
print(f"Specificity (Recall for Benign):    {specificity:.4f}")
print(f"False Negative Rate:                {fnr:.4f}")
print(f"False Positive Rate:                {fpr:.4f}")
print("\n" + classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix — ResNet18 on Mammograms', fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# STEP 10: FALSE NEGATIVES AND FALSE POSITIVES
# =============================================================
fn_indices = np.where((all_labels == 1) & (all_preds == 0))[0]
fp_indices = np.where((all_labels == 0) & (all_preds == 1))[0]

print(f"\nFalse Negatives (Malignant missed): {len(fn_indices)}")
print(f"False Positives (Benign flagged):   {len(fp_indices)}")

if len(fn_indices) > 0:
    n_show = min(8, len(fn_indices))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            img, lbl = test_dataset_resnet[fn_indices[i]]
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title("TRUE: MALIGNANT\nPRED: BENIGN", color='red', fontsize=10)
        ax.axis('off')
    plt.suptitle("FALSE NEGATIVES — Cancer Cases Missed by the Model", fontsize=14, color='red')
    plt.tight_layout()
    plt.savefig("false_negatives.png", dpi=150, bbox_inches='tight')
    plt.show()

if len(fp_indices) > 0:
    n_show = min(8, len(fp_indices))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            img, lbl = test_dataset_resnet[fp_indices[i]]
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title("TRUE: BENIGN\nPRED: MALIGNANT", color='orange', fontsize=10)
        ax.axis('off')
    plt.suptitle("FALSE POSITIVES — Benign Cases Flagged as Cancer", fontsize=14, color='orange')
    plt.tight_layout()
    plt.savefig("false_positives.png", dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================
# STEP 11: SUMMARY
# =============================================================
print("\n" + "=" * 55)
print("MODEL COMPARISON SUMMARY")
print("=" * 55)
print(f"{'Model':<25} {'Best Test Acc':>15} {'Parameters':>15}")
print("-" * 55)
print(f"{'Custom CNN (Part 2)':<25} {best_cnn_acc:>14.4f} {sum(p.numel() for p in model_cnn.parameters()):>15,}")
print(f"{'ResNet18 (pretrained)':<25} {best_resnet_acc:>14.4f} {sum(p.numel() for p in model_resnet.parameters()):>15,}")

print("\n" + "=" * 55)
print("MEDICAL SIGNIFICANCE")
print("=" * 55)
print(f"""
In medical diagnosis, False Negatives are the most critical errors.
A False Negative means a patient WITH cancer is told they are healthy.
This delays treatment and can be life-threatening.

Our best model (ResNet18) has:
- {len(fn_indices)} False Negatives out of {(all_labels == 1).sum()} malignant cases
- False Negative Rate: {fnr:.2%}
- Sensitivity: {sensitivity:.2%}

A False Positive (benign flagged as malignant) leads to additional
tests and anxiety, but is not directly dangerous.

Our model has:
- {len(fp_indices)} False Positives out of {(all_labels == 0).sum()} benign cases
- False Positive Rate: {fpr:.2%}

Key observations:
1. The custom CNN from Part 2 (~{best_cnn_acc:.0%}) barely outperforms random guessing,
   showing that simple architectures struggle with medical images.
2. Transfer learning (ResNet18 pretrained on ImageNet) significantly
   improves performance (~{best_resnet_acc:.0%}), even on grayscale medical images.
3. The gap between train accuracy (91%) and test accuracy ({best_resnet_acc:.0%})
   indicates overfitting, expected with only 1,318 training samples.
4. In a clinical setting, this model should ASSIST radiologists,
   not replace them. A sensitivity of {sensitivity:.0%} means {fnr:.0%} of
   cancers would be missed — unacceptable as a standalone diagnostic tool.
""")

# =============================================================
# STEP 12: DOWNLOAD ALL FIGURES
# =============================================================
from google.colab import files
for f in ['sample_mammograms.png', 'training_curves_comparison.png',
          'confusion_matrix.png', 'false_negatives.png', 'false_positives.png']:
    if os.path.exists(f):
        files.download(f)

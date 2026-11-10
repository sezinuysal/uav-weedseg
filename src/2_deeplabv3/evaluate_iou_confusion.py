# ==============================================================
# ✅ Confusion Matrix ve IoU (Intersection over Union) Hesaplama
# ==============================================================

# =============================
# Standart Kütüphane Importları
# =============================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# PyTorch ve Scikit-learn Importları
# =============================
import torch
from sklearn.metrics import confusion_matrix


def evaluate_model(model, val_ds, device, class_names=None):
    """
    Modelin doğrulama veri seti üzerindeki performansını değerlendirir.

    Bu fonksiyon, confusion matrix ve her sınıf için IoU (Intersection over Union)
    değerlerini hesaplar ve görselleştirir.
    """
    try:
        n_classes = len(class_names) if class_names else 3
        all_preds, all_gts = [], []

        # 1. Validation set üzerindeki tüm örnekler için tahmin al
        model.eval()
        for i in range(len(val_ds)):
            img, mask, _ = val_ds[i]

            # Tensör dönüşümleri
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(mask).long()

            # Tahmin
            with torch.no_grad():
                pred = model(img_tensor)
                pred_classes = pred.argmax(1).squeeze().cpu().numpy()

            # Sonuçları düzleştir
            all_preds.append(pred_classes.flatten())
            all_gts.append(mask_tensor.numpy().flatten())

        # Numpy dizilerine birleştir
        all_preds = np.concatenate(all_preds)
        all_gts = np.concatenate(all_gts)

        # 2. Confusion Matrix Hesapla
        cm = confusion_matrix(all_gts, all_preds, labels=list(range(n_classes)))
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        # 3. Görselleştir (Normalized Heatmap)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names or list(range(n_classes)),
            yticklabels=class_names or list(range(n_classes)),
        )
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix (Normalized)")
        plt.tight_layout()
        plt.show()

        # 4. IoU (Intersection over Union) Hesapla
        ious = []
        for c in range(n_classes):
            tp = cm[c, c]  # True Positive
            fn = cm[c, :].sum() - tp  # False Negative
            fp = cm[:, c].sum() - tp  # False Positive
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0
            ious.append(iou)

        # Sonuçları yazdır
        if class_names:
            for idx, cls_name in enumerate(class_names):
                print(f"IoU {cls_name}: {ious[idx] * 100:.2f}%")
        else:
            for idx in range(n_classes):
                print(f"IoU Class {idx}: {ious[idx] * 100:.2f}%")

        print(f"Mean IoU: {np.mean(ious) * 100:.2f}%")

        # 5. Sonuçları döndür
        return {
            "confusion_matrix": cm,
            "confusion_matrix_normalized": cm_norm,
            "IoUs": ious,
            "mean_IoU": np.mean(ious),
        }

    except Exception as error:
        print(f"⚠️ Model değerlendirme sırasında hata oluştu: {error}")
        return None


if __name__ == "__main__":
    print("Bu modül bir değerlendirme fonksiyonu içerir.")

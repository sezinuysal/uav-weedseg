# =====================================================================
# ✅ Monte Carlo Dropout Değerlendirme (WeedsGalore Dataset, IoU & Entropy)
# =====================================================================

# =============================
# Standart Kütüphane Importları
# =============================
import os
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# =============================
# PyTorch Kütüphane Importları
# =============================
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

# =============================
# ✅ 1. Proje Yolunu Ortama Ekle
# =============================
sys.path.append("/kaggle/input/zip-wg/weedsgalore/src")

# =============================
# ✅ 2. Dataset Modülünü Dinamik Yükle
# =============================
try:
    spec = importlib.util.spec_from_file_location(
        "weedsgalore",
        "/kaggle/input/zip-wg/weedsgalore/src/datasets/weedsgalore/weedsgalore.py",
    )
    weedsgalore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(weedsgalore)
    WeedsGaloreDataset = weedsgalore.WeedsGaloreDataset
    print("✅ WeedsGaloreDataset modülü başarıyla yüklendi.")
except Exception as error:
    print(f"⚠️ Dataset modülü yüklenirken hata oluştu: {error}")
    sys.exit(1)

# =============================
# ✅ 3. Model Import & Yükleme
# =============================
try:
    from nets import deeplabv3plus_resnet50_do

    ckpt_path = "/kaggle/working/out_rgb3_do/last.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = deeplabv3plus_resnet50_do(
        num_classes=3, pretrained_backbone=False
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state["model_state"])
    print(f"✅ Model başarıyla yüklendi: {ckpt_path}")

except Exception as error:
    print(f"⚠️ Model yüklenirken hata oluştu: {error}")
    sys.exit(1)


def enable_mc_dropout(model):
    """
    Modeldeki Dropout katmanlarını aktif bırakır,
    BatchNorm katmanlarını ise eval moduna alır.
    """
    for module in model.modules():
        name = module.__class__.__name__
        if name.startswith("Dropout"):
            module.train()
        elif name.startswith("BatchNorm"):
            module.eval()
        else:
            module.eval()


# ✅ Dropout aktif bırak
enable_mc_dropout(net)

# =============================
# ✅ 4. Dataset Yükleme
# =============================
dataset = WeedsGaloreDataset(
    dataset_path="/kaggle/input/dataset/weedsgalore-dataset",
    dataset_size=10,
    in_bands=3,
    num_classes=3,
    is_training=False,
    split="val",
    augmentation=False,
)

# =============================
# ✅ 5. IoU Değerlendirici
# =============================
evaluator = MulticlassJaccardIndex(num_classes=3, average=None).to(device)


def mc_forward(x, model, mc_samples=5):
    """
    Monte Carlo Dropout ileri besleme fonksiyonu.

    Args:
        x (torch.Tensor): Giriş görüntü tensörü.
        model (torch.nn.Module): Eğitilmiş model.
        mc_samples (int): Dropout örnekleme sayısı.

    Returns:
        tuple:
            - pred (torch.Tensor): Tahmin edilen sınıf maskesi.
            - entropy (torch.Tensor): Belirsizlik (predictive entropy) haritası.
    """
    probs = []
    with torch.no_grad():
        for _ in range(mc_samples):
            logits = model(x)
            p = F.softmax(logits, dim=1)
            probs.append(p.unsqueeze(0))

    probs = torch.cat(probs, dim=0)
    mean_p = probs.mean(dim=0)
    entropy = -(mean_p * mean_p.clamp_min(1e-8).log()).sum(dim=1)
    pred = mean_p.argmax(dim=1)
    return pred, entropy


# =============================
# ✅ 6. Örnek Tahminler & Görselleştirme
# =============================
for idx in range(3):
    features, unique_labels, binary_labels = dataset[idx]
    labels = binary_labels

    x = torch.tensor(features).unsqueeze(0).float().to(device)
    y = torch.tensor(labels).unsqueeze(0).long().to(device)

    pred, entropy = mc_forward(x, net, mc_samples=5)

    # IoU Hesapla
    evaluator.update(pred, y)
    iou = evaluator.compute() * 100
    evaluator.reset()

    # Görselleştirme
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(features, (1, 2, 0)))
    plt.title("Input Image")

    plt.subplot(1, 3, 2)
    plt.imshow(pred.squeeze().cpu().numpy(), cmap="tab20")
    plt.title("Prediction (MC Dropout)")

    plt.subplot(1, 3, 3)
    plt.imshow(entropy.squeeze().cpu().numpy(), cmap="inferno")
    plt.colorbar()
    plt.title("Predictive Entropy")

    plt.suptitle(f"Örnek {idx} | IoU: {iou.cpu().numpy()}", fontsize=12)
    plt.tight_layout()
    plt.show()

# =============================
# ✅ 7. Ortalama IoU Sonucu
# =============================
metrics = evaluator.compute() * 100
print("📊 Ortalama IoU (per class):", metrics.cpu().numpy())

# ================================================================
# setup_all.py
# Tam proje kurulum + konu bazlı commit + (opsiyonel) GitHub push
# Repo: https://github.com/sezinuysal/uav-weedseg
# ================================================================

import os
import subprocess
from textwrap import dedent

# ==========
# Ayarlar
# ==========
ROOT = os.path.abspath("./uav-weedseg")   # Proje kök klasörü
REMOTE = "https://github.com/sezinuysal/uav-weedseg.git"  # GitHub repo
BRANCH = "main"

GROUPS = [
    ("1_env", "chore(env): add torch.load patch & environment config"),
    ("2_deeplabv3", "feat(deeplabv3+): add RGB3 Dropout training & evaluation pipeline"),
    ("3_maskformer_dataset", "feat(maskformer): add WeedsGalore dataset loader & config generator"),
    ("4_maskformer_train", "feat(maskformer): add training & evaluation pipelines (RGB6 + CPU)"),
]

os.makedirs(ROOT, exist_ok=True)
for sub in [g[0] for g in GROUPS]:
    os.makedirs(os.path.join(ROOT, sub), exist_ok=True)

# =========================================================
# 1) ENV: configure_environment.py
# =========================================================
configure_environment_py = dedent("""
    # =============================
    # Standart Kütüphane Importları
    # =============================
    import os

    # =============================
    # PyTorch Kütüphane Importları
    # =============================
    import torch


    def configure_environment():
        """
        PyTorch ortam ayarlarını yapılandırır.

        Bu fonksiyon, 'weights_only' parametresiyle ilgili uyumsuzlukları önlemek için
        torch.load fonksiyonuna güvenli bir yama (monkey patch) uygular ve gerekli
        ortam değişkenlerini ayarlar.
        """
        try:
            # 'weights_only' parametresi için hatalı yüklemeleri önlemek amacıyla çevre değişkeni ayarlanır
            os.environ["TORCH_DISABLE_WEIGHTS_ONLY"] = "1"

            # Orijinal torch.load fonksiyonu kaydedilir
            original_load = torch.load

            def safe_load(*args, **kwargs):
                """
                torch.load fonksiyonunun güvenli sürümüdür.

                Eğer 'weights_only' parametresi verilmemişse, varsayılan olarak False değeri atanır.
                Bu, model ağırlıkları yüklenirken hata riskini azaltır.
                """
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            # Monkey patch işlemi uygulanır
            torch.load = safe_load

            # Python modül yolu (örnek proje yolu) ayarlanır
            os.environ["PYTHONPATH"] = "/kaggle/input/zip-wg/weedsgalore/src"

            print("✅ PyTorch ortamı başarıyla yapılandırıldı.")

        except Exception as error:
            # Hata durumunda bilgilendirici çıktı
            print(f"⚠️ Ortam yapılandırılırken hata oluştu: {error}")


    if __name__ == "__main__":
        # Ortam yapılandırmasını çalıştırır
        configure_environment()
""").lstrip()

# =========================================================
# 1) ENV: train_rgb3_ckpt_runner.py  (ckpt_interval ile)
# =========================================================
train_rgb3_ckpt_runner_py = dedent("""
    # ================================================================
    # ✅ Patch + Train (RGB, 3-Class, DeepLabv3+, ckpt_interval=10)
    # ================================================================

    # =============================
    # Standart Kütüphane Importları
    # =============================
    import os
    import functools

    # =============================
    # PyTorch Kütüphane Importları
    # =============================
    import torch


    def patch_torch_load():
        """
        PyTorch 2.6 için 'weights_only' parametresi uyumluluğu sağlar.

        torch.load fonksiyonunu varsayılan olarak weights_only=False olacak
        şekilde yamalar (monkey patch).
        """
        try:
            torch.load = functools.partial(torch.load, weights_only=False)
            print("✅ torch.load patched (weights_only=False)")
        except Exception as error:
            print(f"⚠️ torch.load patch işlemi başarısız oldu: {error}")


    def set_environment_path():
        """
        PYTHONPATH ortam değişkenini proje kaynak kodu yoluna ayarlar.
        """
        try:
            os.environ["PYTHONPATH"] = "/kaggle/input/zip-wg/weedsgalore/src"
            print("✅ PYTHONPATH başarıyla ayarlandı.")
        except Exception as error:
            print(f"⚠️ Ortam yolu ayarlanırken hata oluştu: {error}")


    def run_training():
        """
        Eğitim sürecini sistem komutu ile başlatır.

        DeepLabv3+ modeli, RGB 3-class veri kümesi üzerinde eğitilir.
        10 iterasyonda bir .pth checkpoint dosyası kaydedilir.
        """
        try:
            command = (
                "python -c 'import torch, functools; "
                "torch.load = functools.partial(torch.load, weights_only=False); "
                "import runpy; "
                "runpy.run_path(\"/kaggle/input/train-py2/train.py\", run_name=\"__main__\")' "
                "--dataset_path /kaggle/input/dataset/weedsgalore-dataset "
                "--ckpt_resnet /kaggle/working/ckpts/resnet50-19c8e357.pth "
                "--in_channels 3 "
                "--num_classes 3 "
                "--epochs 1 "
                "--batch_size 2 "
                "--lr 0.0005 "
                "--dlv3p_do True "
                "--pretrained_backbone True "
                "--out_dir /kaggle/working/out_rgb3_do "
                "--ckpt_interval 10"
            )

            os.system(command)
            print("✅ Eğitim süreci başarıyla tamamlandı.")

        except Exception as error:
            print(f"⚠️ Eğitim süreci başlatılırken hata oluştu: {error}")


    def list_saved_checkpoints():
        """Kaydedilen checkpoint (.pth) dosyalarını listeler."""
        try:
            print("📂 Kaydedilen dosyalar:")
            os.system("ls -lh /kaggle/working/out_rgb3_do")
        except Exception as error:
            print(f"⚠️ Kayıtlı dosyalar listelenirken hata oluştu: {error}")


    def main():
        """
        Ana çalışma fonksiyonudur.

        Ortam yamalarını uygular, eğitimi başlatır ve
        kaydedilen checkpoint dosyalarını listeler.
        """
        patch_torch_load()
        set_environment_path()
        run_training()
        list_saved_checkpoints()


    if __name__ == "__main__":
        main()
""").lstrip()

# =========================================================
# 2) DEEPLABV3+: evaluate_vimc_runner.py (CLI çağrı)
# =========================================================
evaluate_vimc_runner_py = dedent("""
    import os

    def main():
        """
        evaluate_vimc.py betiğini uygun parametrelerle çalıştırır.
        """
        command = (
            "python /kaggle/input/evaluate-vimc-py/evaluate_vimc.py "
            "--dataset_path /kaggle/input/dataset/weedsgalore-dataset "
            "--ckpt /kaggle/working/last_rgb3_do_model_only.pth "
            "--in_channels 5 "
            "--num_classes 3 "
            "--split test "
            "--mc_samples 5"
        )
        os.system(command)

    if __name__ == "__main__":
        main()
""").lstrip()

# =========================================================
# 2) DEEPLABV3+: mc_dropout_eval.py
# =========================================================
mc_dropout_eval_py = dedent("""
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
""").lstrip()

# =========================================================
# 2) DEEPLABV3+: evaluate_iou_confusion.py
# =========================================================
evaluate_iou_confusion_py = dedent("""
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
""").lstrip()

# =========================================================
# 2) DEEPLABV3+: wandb_val_upload.py
# =========================================================
wandb_val_upload_py = dedent("""
    # ============================================================
    # ✅ WeedsGalore Validation Görsellerinin W&B'ye Loglanması
    # ============================================================

    # =============================
    # Kütüphane Importları
    # =============================
    import wandb
    import torch


    def log_validation_predictions(model, val_ds, project_name, entity_name, run_name):
        """
        Modelin ilk 5 doğrulama örneğini tahminleriyle birlikte W&B’ye loglar.
        """
        try:
            # 1. W&B Oturum Başlat
            wandb.init(
                project=project_name,
                entity=entity_name,
                name=run_name,
            )

            # 2. Cihaz Ayarı
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()

            class_labels = {0: "Background", 1: "Weed", 2: "Crop"}
            logged_images = []

            # 3. İlk 5 Validation Örneğini İşle
            for i in range(5):
                img, mask, _ = val_ds[i]  # numpy array döner (C, H, W)

                # Tensör dönüşümleri
                img_tensor = torch.from_numpy(img).float()
                mask_tensor = torch.from_numpy(mask).long()

                # Batch boyutu ekle ve GPU’ya gönder
                img_batch = img_tensor.unsqueeze(0).to(device)

                # Model tahmini
                with torch.no_grad():
                    pred = model(img_batch)
                    pred_classes = pred.argmax(1).squeeze().cpu().numpy()

                # Görseli normalize et (0–1)
                img_disp = (
                    img_tensor.permute(1, 2, 0).cpu().numpy()
                    - img_tensor.min()
                ) / (img_tensor.max() - img_tensor.min() + 1e-8)

                # W&B için görsel ve maskeleri hazırla
                wandb_image = wandb.Image(
                    img_disp,
                    caption=f"Sample {i}",
                    masks={
                        "ground_truth": {
                            "mask_data": mask_tensor.cpu().numpy(),
                            "class_labels": class_labels,
                        },
                        "prediction": {
                            "mask_data": pred_classes,
                            "class_labels": class_labels,
                        },
                    },
                )

                logged_images.append(wandb_image)

            # 4. W&B'ye Logla
            wandb.log({"val_predictions": logged_images})
            wandb.finish()
            print("✅ İlk 5 validation örneği W&B’ye başarıyla yüklendi!")

        except Exception as error:
            print(f"⚠️ W&B loglama sırasında hata oluştu: {error}")


    if __name__ == "__main__":
        print("Bu modül bir W&B log fonksiyonu içerir.")
""").lstrip()

# =========================================================
# 3) MASKFORMER DATASET: maskformer_dataset_loader.py
# (Detectron2 kayıt + loader + converter)
# =========================================================
maskformer_dataset_loader_py = dedent("""
    import sys
    import os
    import glob
    import numpy as np
    from PIL import Image
    import torch
    from detectron2.data import DatasetCatalog, MetadataCatalog

    class WeedsGaloreDatasetLoader:
        """
        WeedsGalore dataset yapısına özel loader.
        Yapı: dataset/YYYY-MM-DD/images/*.png ve semantics/*.png
        """
        def __init__(self, dataset_path, split="train", in_bands=3, num_classes=3, 
                     is_training=True, augmentation=False, dataset_size=None):
            self.dataset_path = dataset_path
            self.split = split
            self.in_bands = in_bands
            self.num_classes = num_classes
            self.is_training = is_training
            self.augmentation = augmentation
            self.dataset_size = dataset_size
            
            if in_bands == 3:
                self.bands = ['R', 'G', 'B']
            elif in_bands == 5:
                self.bands = ['B', 'G', 'R', 'RE', 'NIR']
            else:
                raise ValueError(f"in_bands {in_bands} desteklenmiyor. 3 veya 5 olmalı.")
            
            self.data = self._load_dataset()
            self.length = len(self.data)
            print(f"✅ {split} dataset: {self.length} örnek yüklendi")
            
        def _load_dataset(self):
            """Dataset'teki tüm örnekleri yükle"""
            date_folders = glob.glob(os.path.join(self.dataset_path, "20*"))
            date_folders = sorted(date_folders)
            if not date_folders:
                print(f"❌ HATA: {self.dataset_path} içinde tarih klasörleri bulunamadı!")
                return []
            samples = []
            for date_folder in date_folders:
                date_name = os.path.basename(date_folder)
                images_dir = os.path.join(date_folder, "images")
                semantics_dir = os.path.join(date_folder, "semantics")
                if not os.path.exists(images_dir) or not os.path.exists(semantics_dir):
                    continue
                image_files = glob.glob(os.path.join(images_dir, "*.png"))
                sample_ids = {os.path.basename(p).rsplit('_', 1)[0] for p in image_files}
                for sid in sorted(sample_ids):
                    band_paths = {}
                    all_exist = True
                    for band in ['B', 'G', 'R', 'RE', 'NIR']:
                        pth = os.path.join(images_dir, f"{sid}_{band}.png")
                        if os.path.exists(pth):
                            band_paths[band] = pth
                        else:
                            all_exist = False
                            break
                    if not all_exist:
                        continue
                    sem_p = os.path.join(semantics_dir, f"{sid}.png")
                    if not os.path.exists(sem_p):
                        sem_p = os.path.join(semantics_dir, f"{sid}_semantic.png")
                        if not os.path.exists(sem_p):
                            continue
                    samples.append({
                        'sample_id': sid,
                        'band_paths': band_paths,
                        'semantic_path': sem_p,
                        'date': date_name
                    })
            if len(samples) == 0:
                print("❌ HİÇ ÖRNEK BULUNAMADI! Dosya yapısını kontrol edin.")
                return []
            np.random.seed(42)
            indices = np.random.permutation(len(samples))
            split_idx = int(len(samples) * 0.8)
            if self.split == "train":
                selected_indices = indices[:split_idx]
            elif self.split == "val":
                selected_indices = indices[split_idx:]
            else:
                selected_indices = indices
            samples = [samples[i] for i in selected_indices]
            if self.dataset_size is not None and self.dataset_size < len(samples):
                samples = samples[:self.dataset_size]
            return samples
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            """Bir örneği yükle ve döndür"""
            sample = self.data[idx]
            channels = []
            for band in self.bands:
                img = Image.open(sample['band_paths'][band])
                img = np.array(img, dtype=np.float32)
                channels.append(img)
            image = np.stack(channels, axis=-1)
            image = image / 255.0
            mask = Image.open(sample['semantic_path'])
            mask = np.array(mask, dtype=np.int64)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask).long()
            info = {
                'file_name': sample['sample_id'] + '.png',
                'height': image.shape[0],
                'width': image.shape[1],
                'sample_id': sample['sample_id'],
                'date': sample['date']
            }
            return image_tensor, mask_tensor, info

    def convert_to_dict(ds, prefix="train"):
        """Dataset'i Detectron2 formatına çevir"""
        if ds is None or len(ds) == 0:
            print(f"❌ {prefix} dataset boş!")
            return []
        dataset_dicts = []
        for i in range(len(ds)):
            try:
                features, labels, info = ds[i]
                dataset_dicts.append({
                    "file_name": info['file_name'],
                    "image_id": i,
                    "height": info['height'],
                    "width": info['width'],
                    "image": features.permute(1, 2, 0).numpy(),
                    "sem_seg": labels.numpy(),
                })
            except Exception as e:
                print(f"⚠️  {prefix}[{i}] hata: {e}")
                continue
        print(f"✅ {prefix}: {len(dataset_dicts)} örnek hazır!")
        return dataset_dicts

    def weeds_train():
        ds = WeedsGaloreDatasetLoader(
            dataset_path="/kaggle/input/dataset/weedsgalore-dataset",
            split="train",
            in_bands=3,
            num_classes=3,
            is_training=True,
            augmentation=True,
            dataset_size=None
        )
        return convert_to_dict(ds, "train")

    def weeds_val():
        ds = WeedsGaloreDatasetLoader(
            dataset_path="/kaggle/input/dataset/weedsgalore-dataset",
            split="val",
            in_bands=3,
            num_classes=3,
            is_training=False,
            augmentation=False,
            dataset_size=None
        )
        return convert_to_dict(ds, "val")

    def register_datasets():
        """Detectron2 kayıtlarını yapar."""
        for d in ["weedsgalore_train", "weedsgalore_val"]:
            if d in DatasetCatalog.list():
                DatasetCatalog.remove(d)
                MetadataCatalog.remove(d)
        DatasetCatalog.register("weedsgalore_train", weeds_train)
        MetadataCatalog.get("weedsgalore_train").set(
            stuff_classes=["background", "crop", "weed"],
            evaluator_type="sem_seg",
            thing_classes=["crop", "weed"]
        )
        DatasetCatalog.register("weedsgalore_val", weeds_val)
        MetadataCatalog.get("weedsgalore_val").set(
            stuff_classes=["background", "crop", "weed"],
            evaluator_type="sem_seg",
            thing_classes=["crop", "weed"]
        )
        print("✅ Detectron2 dataset kayıtları tamamlandı.")
""").lstrip()

# =========================================================
# 3) MASKFORMER DATASET: save_maskformer_config.py
# =========================================================
save_maskformer_config_py = dedent("""
    # ===========================================================
    # ✅ MaskFormer Konfigürasyon YAML Dosyasını Yeniden Kaydetme
    # ===========================================================

    import os

    def save_maskformer_yaml(output_path):
        """
        MaskFormer modelinin eğitim konfigürasyonunu YAML dosyası olarak kaydeder.
        """
        try:
            yaml_content = \"\"\"MODEL:
      META_ARCHITECTURE: "MaskFormer"
      WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
      RESNETS:
        DEPTH: 50
      SEM_SEG_HEAD:
        NAME: "MaskFormerHead"
        IGNORE_VALUE: 255
        NUM_CLASSES: 3
        LOSS_WEIGHT: 1.0
      MASK_FORMER:
        TRANSFORMER_DECODER_NAME: "MultiScaleMaskedTransformerDecoder"
        HIDDEN_DIM: 256
        NUM_OBJECT_QUERIES: 100
        NHEADS: 8
        DROPOUT: 0.1
        DIM_FEEDFORWARD: 2048
        ENC_LAYERS: 0
        DEC_LAYERS: 6
        PRE_NORM: False
        DEEP_SUPERVISION: True
        NO_OBJECT_WEIGHT: 0.1
    INPUT:
      FORMAT: "RGB"
    SOLVER:
      IMS_PER_BATCH: 4
      BASE_LR: 0.00003
      MAX_ITER: 160000
      STEPS: (120000, 140000)
      CHECKPOINT_PERIOD: 5000
    DATASETS:
      TRAIN: ("weedsgalore_train",)
      TEST: ("weedsgalore_val",)
    DATALOADER:
      NUM_WORKERS: 4
    OUTPUT_DIR: "./output"
    \"\"\"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(yaml_content)
            print(f"✅ YAML dosyası başarıyla kaydedildi: {output_path}")
        except Exception as error:
            print(f"⚠️ YAML dosyası kaydedilirken hata oluştu: {error}")


    if __name__ == "__main__":
        save_maskformer_yaml("/kaggle/working/maskformer_R50_bs16_160k.yaml")
""").lstrip()

# =========================================================
# 4) MASKFORMER TRAIN: train_maskformer_cpu.py
# =========================================================
train_maskformer_cpu_py = dedent("""
    # ===============================================================
    # ✅ MaskFormer CPU Eğitim Betiği (Detectron2 + Güvenli Yükleme)
    # ===============================================================

    import sys
    import os
    import glob
    import torch
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.config import get_cfg
    from detectron2.utils.logger import setup_logger

    try:
        from mask_former.config import add_mask_former_config
    except ImportError:
        print("⚠️  'mask_former.config' modülü bulunamadı.")
        def add_mask_former_config(cfg):
            print("⚠️  add_mask_former_config boş bırakıldı.")
            pass

    try:
        from detectron2.projects.deeplab import add_deeplab_config
    except ImportError:
        print("⚠️  'detectron2.projects.deeplab' modülü bulunamadı.")
        def add_deeplab_config(cfg):
            print("⚠️  add_deeplab_config boş bırakıldı.")
            pass

    try:
        from train_net import Trainer
    except ImportError:
        print("❌ 'train_net.py' bulunamadı. Trainer sınıfı yüklenemedi.")
        sys.exit(1)


    def get_dicts():
        """Eğitim veri setini Detectron2 formatında döndürür."""
        data_dir = "/tmp/"
        image_pattern = os.path.join(data_dir, "final_*.png")
        files = sorted(glob.glob(image_pattern))
        if not files:
            print(f"❌ '{image_pattern}' altında resim bulunamadı.")
            return []
        dataset = []
        for f in files:
            dataset.append({
                "file_name": f,
                "sem_seg_file_name": f.replace("final_", "finalmask_"),
                "height": 600,
                "width": 600,
                "image_id": os.path.basename(f)
            })
        print(f"✅ {len(dataset)} örnek başarıyla yüklendi.")
        return dataset


    def register_dataset():
        """Detectron2 dataset kaydını gerçekleştirir."""
        if "train" in DatasetCatalog.list():
            DatasetCatalog.remove("train")
            MetadataCatalog.remove("train")
        dataset_dicts = get_dicts()
        if not dataset_dicts:
            sys.exit("❌ Veri seti yüklenemedi.")
        DatasetCatalog.register("train", lambda: dataset_dicts)
        MetadataCatalog.get("train").set(
            stuff_classes=["bg", "crop", "weed"],
            ignore_label=255,
            evaluator_type="sem_seg"
        )
        print("✅ Dataset Detectron2’ye başarıyla kaydedildi.")


    def setup_config():
        """Eğitim konfigürasyonunu ayarlar ve dondurur."""
        cfg = get_cfg()
        add_mask_former_config(cfg)
        add_deeplab_config(cfg)
        config_path = "/kaggle/working/maskformer_wg.yaml"
        if not os.path.exists(config_path):
            sys.exit(f"❌ Yapılandırma dosyası bulunamadı: {config_path}")
        cfg.merge_from_file(config_path)
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATASETS.TEST = ("train",)
        cfg.SOLVER.MAX_ITER = 20
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
        try:
            cfg.MODEL.MASK_FORMER.MASK_DIM = 256
            print("✅ mask_dim: MODEL.MASK_FORMER altında tanımlandı.")
        except AttributeError:
            try:
                cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
                print("✅ mask_dim: MODEL.SEM_SEG_HEAD altında tanımlandı.")
            except AttributeError:
                print("⚠️  mask_dim parametresi bulunamadı, varsayılan değer kullanılacak.")
        cfg.MODEL.DEVICE = "cpu"
        cfg.OUTPUT_DIR = "/kaggle/working/maskformer_out"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        try:
            cfg.freeze()
            print("✅ Konfigürasyon başarıyla donduruldu.")
        except Exception as error:
            sys.exit(f"❌ Konfigürasyon dondurulamadı: {error}")
        return cfg


    def run_training(cfg):
        """Trainer sınıfını kullanarak modeli eğitir."""
        setup_logger()
        print("🚀 Trainer oluşturuluyor...")
        try:
            trainer = Trainer(cfg)
            print("✅ Trainer başarıyla oluşturuldu.")
        except Exception as error:
            sys.exit(f"❌ Trainer oluşturulamadı: {error}")
        try:
            trainer.resume_or_load(resume=False)
            print("✅ Model yüklendi, eğitim başlıyor...")
        except Exception as error:
            sys.exit(f"❌ Model yüklenemedi: {error}")
        try:
            trainer.train()
            print("✅ Eğitim tamamlandı.")
        except Exception as error:
            sys.exit(f"❌ Eğitim sırasında hata oluştu: {error}")


    if __name__ == "__main__":
        print("\\n=== MASKFORMER CPU EĞİTİM ===\\n")
        register_dataset()
        cfg = setup_config()
        run_training(cfg)
        print("\\n✅ TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI.")
""").lstrip()

# =========================================================
# 4) MASKFORMER TRAIN: train_maskformer_rgb6.py
# =========================================================
train_maskformer_rgb6_py = dedent("""
    # ============================================================
    # ✅ MaskFormer RGB (6 Class, res4) Eğitim Betiği
    # ============================================================

    import os
    import sys
    import runpy
    from detectron2.data import MetadataCatalog


    def check_metadata(dataset_name):
        """Detectron2 Metadata kontrolünü yapar."""
        try:
            meta = MetadataCatalog.get(dataset_name)
            print(f"✅ Metadata bulundu: {dataset_name}")
            print("   Sınıflar:", meta.stuff_classes)
        except KeyError:
            print(f"❌ Hata: '{dataset_name}' isimli metadata bulunamadı.")
            sys.exit(1)


    def write_config_file(cfg_path):
        """MaskFormer eğitim konfigürasyon dosyasını (YAML) oluşturur."""
        try:
            yaml_content = \"\"\"MODEL:
      META_ARCHITECTURE: "MaskFormer"
      WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
      SEM_SEG_HEAD:
        NAME: "MaskFormerHead"
        IGNORE_VALUE: 255
        NUM_CLASSES: 6
        IN_FEATURES: ["res2", "res3", "res4", "res5"]
      MASK_FORMER:
        TRANSFORMER_IN_FEATURE: "res4"

    INPUT:
      FORMAT: "BGR"

    SOLVER:
      IMS_PER_BATCH: 2
      BASE_LR: 0.00003
      MAX_ITER: 200
      CHECKPOINT_PERIOD: 100

    DATASETS:
      TRAIN: ("weedsgalore_rgb6",)
      TEST: ()

    DATALOADER:
      NUM_WORKERS: 2

    OUTPUT_DIR: "/kaggle/working/maskformer_rgb6_out"
    \"\"\"
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as file:
                file.write(yaml_content.strip() + "\\n")
            print(f"✅ Konfigürasyon dosyası başarıyla kaydedildi: {cfg_path}")
        except Exception as error:
            print(f"❌ Config dosyası oluşturulamadı: {error}")
            sys.exit(1)


    def run_training(cfg_path):
        """MaskFormer eğitim sürecini başlatır."""
        try:
            sys.argv = [
                "train_net.py",
                "--config-file", cfg_path,
                "--num-gpus", "1",
                "OUTPUT_DIR", "/kaggle/working/maskformer_rgb6_out",
                "MODEL.WEIGHTS", "detectron2://ImageNetPretrained/MSRA/R-50.pkl",
                "SOLVER.MAX_ITER", "200",
                "SOLVER.IMS_PER_BATCH", "2",
                "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "6",
                "MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE", "res4"
            ]
            print("🚀 Eğitim başlatılıyor (RGB, 6 sınıf, res4 ile)...")
            runpy.run_path(
                "/kaggle/input/kaggle-work/MaskFormer-main/train_net.py",
                run_name="__main__"
            )
            print("✅ Eğitim başarıyla tamamlandı!")
        except FileNotFoundError:
            print("❌ train_net.py dosyası bulunamadı.")
            sys.exit(1)
        except Exception as error:
            print(f"❌ Eğitim sırasında hata oluştu: {error}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


    def main():
        """Ana fonksiyon: metadata kontrolü, config oluşturma, eğitimi başlatma."""
        dataset_name = "weedsgalore_rgb6"
        cfg_path = "/kaggle/working/rgb6.yaml"
        check_metadata(dataset_name)
        write_config_file(cfg_path)
        run_training(cfg_path)


    if __name__ == "__main__":
        main()
""").lstrip()

# =========================================================
# Dosyaları yaz
# =========================================================
files_to_write = [
    ("1_env/configure_environment.py", configure_environment_py),
    ("1_env/train_rgb3_ckpt_runner.py", train_rgb3_ckpt_runner_py),
    ("2_deeplabv3/evaluate_vimc_runner.py", evaluate_vimc_runner_py),
    ("2_deeplabv3/mc_dropout_eval.py", mc_dropout_eval_py),
    ("2_deeplabv3/evaluate_iou_confusion.py", evaluate_iou_confusion_py),
    ("2_deeplabv3/wandb_val_upload.py", wandb_val_upload_py),
    ("3_maskformer_dataset/maskformer_dataset_loader.py", maskformer_dataset_loader_py),
    ("3_maskformer_dataset/save_maskformer_config.py", save_maskformer_config_py),
    ("4_maskformer_train/train_maskformer_cpu.py", train_maskformer_cpu_py),
    ("4_maskformer_train/train_maskformer_rgb6.py", train_maskformer_rgb6_py),
]

for rel, content in files_to_write:
    path = os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print(f"✅ Tüm dosyalar yazıldı -> {ROOT}")

# =========================================================
# Git init + topic bazlı commit
# =========================================================
def sh(cmd):
    return subprocess.run(cmd, check=True)

os.chdir(ROOT)

# init
sh(["git", "init"])
sh(["git", "branch", "-M", BRANCH])

# staged initial add for all files (so history starts clean)
sh(["git", "add", "."])
sh(["git", "commit", "-m", "build: scaffold repository structure"])

# topic commits
for folder, msg in GROUPS:
    sh(["git", "add", folder])
    sh(["git", "commit", "-m", msg])

print("✅ Lokal commitler tamamlandı.")

# =========================================================
# (Opsiyonel) GitHub push
# =========================================================
try:
    sh(["git", "remote", "remove", "origin"])
except Exception:
    pass

try:
    sh(["git", "remote", "add", "origin", REMOTE])
    sh(["git", "push", "-u", "origin", BRANCH])
    print("🚀 GitHub'a push tamam!")
except Exception as e:
    print("⚠️ GitHub push başarısız. Muhtemel nedenler:")
    print("- Repo erişim yetkisi yok / token gerekli")
    print("- Repo mevcut değil veya boş değil")
    print("- Ağ/SSL kısıtları")
    print("Yine de yerel repo ve commitler hazır.")
    print("Manuel push için:")
    print(f"  cd {ROOT} && git remote add origin {REMOTE} && git push -u origin {BRANCH}")

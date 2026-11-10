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

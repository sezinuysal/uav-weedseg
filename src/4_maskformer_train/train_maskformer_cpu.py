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

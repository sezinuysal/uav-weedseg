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
        yaml_content = """MODEL:
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
"""
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

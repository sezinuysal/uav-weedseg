# ===========================================================
# ✅ MaskFormer Konfigürasyon YAML Dosyasını Yeniden Kaydetme
# ===========================================================

import os

def save_maskformer_yaml(output_path):
    """
    MaskFormer modelinin eğitim konfigürasyonunu YAML dosyası olarak kaydeder.
    """
    try:
        yaml_content = """MODEL:
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
"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(yaml_content)
        print(f"✅ YAML dosyası başarıyla kaydedildi: {output_path}")
    except Exception as error:
        print(f"⚠️ YAML dosyası kaydedilirken hata oluştu: {error}")


if __name__ == "__main__":
    save_maskformer_yaml("/kaggle/working/maskformer_R50_bs16_160k.yaml")

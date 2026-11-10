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

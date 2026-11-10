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

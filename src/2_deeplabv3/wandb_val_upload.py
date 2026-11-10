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

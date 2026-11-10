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

from pathlib import Path
from ultralytics import YOLO


def main():
    # -----------------------------
    # Paths
    # -----------------------------
    data_yaml = "/data_64T_1/Barathi/Projects/SRC_detection/digestpath_dataset/digestpath_yolo/digestpath.yaml"
    project_dir = "/home/barathi1/Project/11.SRC_detection/runs_yolo26"
    run_name = "digestpath_yolo26l_1024"

    # -----------------------------
    # Model and training config
    # -----------------------------
    model_name = "models/yolo26l.pt"   # auto-downloads if missing
    imgsz = 1024
    epochs = 100
    batch = 8
    device = 0
    workers = 16
    seed = 42

    # -----------------------------
    # Train
    # -----------------------------
    model = YOLO(model_name)

    model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        project=project_dir,
        name=run_name,
        pretrained=True,
        cache=True,
        seed=seed,
        patience=20,
        cos_lr=True,
        optimizer="auto",
        amp=True,
        exist_ok=True,
         )

    # -----------------------------
    # Best checkpoint path
    # -----------------------------
    best_ckpt = Path(project_dir) / run_name / "weights" / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    # -----------------------------
    # Validate
    # -----------------------------
    best_model = YOLO(str(best_ckpt))
    best_model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
    )

    # -----------------------------
    # Predict on validation images
    # -----------------------------
    val_source = "/data_64T_1/Barathi/Projects/SRC_detection/digestpath_dataset/digestpath_yolo/images/val"
    best_model.predict(
        source=val_source,
        imgsz=imgsz,
        conf=0.25,
        device=device,
        save=True,
        project=project_dir,
        name=f"{run_name}_pred_val",
        exist_ok=True,
    )

    print(f"\nDone.\nBest checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
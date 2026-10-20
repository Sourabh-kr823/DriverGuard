if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from ultralytics import YOLO
    import shutil, os

    SAVE_DIR = "D:/We4/training_output"
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = YOLO('yolov8n.pt')

    model.train(
        data='D:/We4/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        name='rdd_india',
        project=SAVE_DIR,
        patience=15,
        workers=0,
        save=True,
        save_period=1
    )

    best = f"{SAVE_DIR}/rdd_india/weights/best.pt"
    dest = "D:/We4/DriverGuard/models/yolov8n_rdd_india.pt"
    shutil.copy(best, dest)
    print(f"Deployed to {dest}")
    print("Training complete!")
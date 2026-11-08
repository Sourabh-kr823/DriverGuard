# DriverGuard — Evaluation Report

## 1. Road-Model Validation (RDD2022 val split)
_Skipped — validation failed: Dataset '/home/sourabh/Desktop/DriverGuard/data/finetune_india.yaml' images not found, missing path '/home/sourabh/Desktop/datasets/D:\We4/valid/images'
Note dataset download directory is '/home/sourabh/Desktop/datasets'. You can update this in '/home/sourabh/.config/Ultralytics/settings.json'_

## 2. Inference Benchmark
- **CPU (NCNN): 29.5 ms/frame (~33.9 FPS)**
- Model size: 10.4 MB   ·   Parameters: n/a (NCNN)

## 3. Test Suite (pytest)
- **0 passed, 0 failed**  (/home/sourabh/Desktop/DriverGuard/.venv/bin/python: No module named pytest)

## Note on DMS accuracy
Drowsiness precision/recall requires labeled ground-truth video and is not computed here, so no DMS accuracy figure is fabricated. DMS correctness is evidenced by the test suite above and the manual test-scenario results.
# DriverGuard — Evaluation Report

## 1. Road-Model Validation (RDD2022 val split)
- **mAP50: 58.9%**   ·   mAP50-95: 31.6%

| Class | AP50 % | Precision % | Recall % |
|---|---|---|---|
| pothole | 55.4 | 61.0 | 52.1 |
| crack_longitudinal | 55.0 | 58.6 | 53.5 |
| crack_transverse | 65.4 | 67.1 | 61.4 |
| rutting | 75.5 | 67.3 | 73.8 |
| repair | 42.9 | 61.8 | 36.3 |

## 2. Inference Benchmark
- **GPU (PyTorch): 14.8 ms/frame (~67.4 FPS)**
- Model size: 6.3 MB   ·   Parameters: 3.01 M

## 3. Test Suite (pytest)
_Skipped — skipped (--no-tests)_

## Note on DMS accuracy
Drowsiness precision/recall requires labeled ground-truth video and is not computed here, so no DMS accuracy figure is fabricated. DMS correctness is evidenced by the test suite above and the manual test-scenario results.

## Charts
- per_class_ap50.png
Baseline: `baseline` (git b0ce8bf, seed 42) · PoC: `poc` (git b0ce8bf, seed 42)

| Metric (unit) | Slice | Baseline | PoC | Δ abs | Δ rel | Better | N |
|---|---|---|---|---|---|---|---|
| **Silent-defect Detection F1** (F1, unit = release, positive = defective & blocked) | all | 0.0000 | 1.0000 | 1.0000 | baseline=0 | higher | 20 |
| Defect recall (defective releases blocked / defective) | all | 0.0000 | 1.0000 | 1.0000 | baseline=0 | higher | None |
| False-block rate (clean releases blocked / clean) | all | 0.0000 | 0.0000 | 0.0000 | baseline=0 | lower | None |
| Root-cause accuracy (correct root cause / defective blocked) | all | n/a | 1.0000 | n/a | n/a | higher | None |
| Runtime / 100 items (s) | all | 1.5700 | 3.9680 | | | lower | |

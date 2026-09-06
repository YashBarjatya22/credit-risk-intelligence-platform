# Analysis walkthrough

The authoritative reproducible analysis is `train.py`; it performs every data split, EDA output, model comparison, calibration, threshold selection, rule extraction and final test evaluation in one deterministic run.

To keep the submitted repository lightweight and avoid duplicating a 166 MB input inside a notebook, open a terminal at the project root and run:

```sh
python train.py --data data/application_train.csv
```

Generated outputs appear in `reports/` and `models/`. The final holdout is used only after model selection, calibration, threshold and bands are fixed. See the main README for exact results and limitations.

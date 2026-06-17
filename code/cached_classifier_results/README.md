# Cached classifier results

`gp_classification_predictions.csv` holds the **results of applying a trained classifier to the data**, not the classifier model itself.

## What produced it

During an earlier run of notebook 4 (`FINAL_4_manually_proofed_Ai65_stats_regression.ipynb`), a **Gaussian Process classifier** (RBF kernel, multi-class one-vs-rest) was trained to predict the injection region of each ipsilateral LC soma from its CCF coordinates (`RC, DV, ML`). This file captures that classifier's output after it was applied back to the cells: the predicted region, the per-region posterior probabilities, and a per-cell uncertainty.

We deliberately store these **applied results** rather than the trained model. The fitted GP is large and its serialized form is tied to specific library versions, whereas this CSV is small, human-readable, and version-independent. The model itself is never re-invoked downstream (only these outputs are used).

## Columns (one row per ipsilateral soma)

| Column | Meaning |
|---|---|
| `RC`, `DV`, `ML` | CCF coordinates (microns) — the classifier's input features |
| `true_region` | the actual injection region for that cell |
| `predicted_region` | region predicted by the GP classifier |
| `entropy` | prediction uncertainty (Shannon entropy of the class probabilities) |
| `p_CB` … `p_iontoTH` | the GP's posterior probability for each region |

## Regenerating it (training a new classifier)

Notebook 4 loads this file by default. To train a **new** classifier instead, set the flag near the top of the Gaussian-Process cell:

```python
USE_CACHED_CLASSIFIER_RESULTS = False   # default is True
```

With the flag `False`, the notebook retrains the GP from scratch (**expensive — on the order of hours**), applies it to the data, and overwrites this CSV with the **same columns and format** as the current file — so the retrain branch doubles as the recipe for how this file is produced. Leave it `True` for normal runs.

Note: a regenerated file reproduces the same *methodology*, but its values need not match this file exactly. GP fitting is not guaranteed to be reproducible across library versions (which is why the results are cached). The committed CSV is the authoritative frozen version.

# Project Explained: How This Works and Why It's Built This Way

This document explains the project in plain language: what each file does, how they connect, and — for every non-obvious choice — *why we did it that way instead of the obvious alternative*. Read `README.md` first if you just want to run things; read this if you want to actually understand or explain the project.

---

## 1. What this project is, in one paragraph

We have 10,000 past records of a milling machine's readings (temperature, speed, torque, tool wear) and whether it failed or not. Only 3.4% actually failed — failures are rare. We train a model to look at a machine's current readings and predict "will this fail?" We try three different modeling approaches, pick the best one, and squeeze extra performance out of it with automated tuning.

---

## 2. The data, and the one trap we had to avoid

The raw dataset has these columns:

| Column | Meaning |
|---|---|
| `Type` | Product quality: Low / Medium / High |
| `Air_Temperature`, `Process_Temperature` | Sensor readings, in Kelvin |
| `Rotational_Speed`, `Torque`, `Tool_Wear` | Sensor readings |
| `Machine_Failure` | **The answer** — did it fail? (0/1) |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | *Which specific failure mode* caused the failure |

**Why we never use `TWF`/`HDF`/`PWF`/`OSF`/`RNF` as inputs:** these columns basically say "yes, and here's specifically why it failed." Giving them to the model as inputs would be like asking someone to predict who won a race, but also handing them the race results — the model would just read the answer off these columns instead of learning from the real sensor data. This is called **target leakage**, and it would make the model look great in testing while being useless on a truly new, unseen machine. `src/config.py` lists them under `LEAKAGE_COLUMNS` specifically so every other file can check against one shared list rather than each file re-deciding this.

---

## 3. How the code is organized, and why it's split this way

Instead of one giant script (which is how the original project started, as a single Colab notebook), the logic is split into small files that each do one job:

```
src/config.py             -> all shared settings in one place (paths, random seed, feature lists)
src/logger.py             -> one shared way to print + save log messages
src/data/preprocessing.py -> load data, clean it, split it into train/test
src/visualization/plots.py-> every chart, in one place
src/models/train.py       -> build and train the 3 models
src/models/evaluate.py    -> score the models, generate reports/charts
src/models/tune.py        -> automatically try many settings on the best model
src/models/predict.py     -> use the finished model on one new reading
src/run_pipeline.py       -> runs everything above, in order, in one command
```

**Why split it up instead of one script?** Two reasons:
1. **You can run and inspect one stage at a time** — e.g. run just `python -m src.models.train` to check training works, without waiting for the whole pipeline. Useful for debugging and for a faculty walkthrough where you want to show one piece at a time.
2. **No duplicated logic.** The notebook, the standalone scripts, and the full pipeline all call the *exact same functions*. If `plot_confusion_matrix()` only existed duplicated in three places, fixing a bug would mean fixing it three times (and probably forgetting one).

**Why `config.py` specifically:** things like "where is the dataset file" or "what random seed do we use" are needed in almost every other file. If each file hardcoded its own copy, changing one thing later (e.g. moving the dataset folder) would mean hunting through every file. Instead, every other file does `from src.config import ...` and there's exactly one place to change.

---

## 4. Step by step: what happens when you run `python -m src.run_pipeline`

This is the order of operations, and the reasoning behind that order:

1. **Load the raw CSV** (`load_data`) — reads the file, logs how many rows/columns, checks for missing values and duplicates. You always want to check data quality *before* doing anything else with it.
2. **Clean it** (`clean_data`) — renames columns to Python-friendly names (`Air temperature [K]` → `Air_Temperature`), and separates out the leakage columns.
3. **Generate EDA (exploratory) charts** — failure distribution, correlations, boxplots, etc. This happens *before* modeling because you should understand your data before you try to model it — otherwise you might miss obvious problems (e.g. if 50% of values were missing, you'd want to know before training anything).
4. **Split into train/test** (`split_data`) — 80% of the data to train on, 20% held back to test on, **before any model touches it**. This split happens early and everything downstream respects it, because testing a model on data it already learned from would make it look artificially good — like grading a student on the exact questions you gave them as homework answers.
   - We use a **stratified** split, meaning both the training set and test set keep the same ~3.4% failure rate. Without this, a random split might accidentally put almost all the rare failure cases into one side.
5. **Train 3 models** (`train.py`) — Logistic Regression, Decision Tree, Random Forest. We try multiple approaches because you don't know in advance which one fits this particular data best — it's genuinely an experiment, not something you can predict without trying.
6. **Evaluate all 3** (`evaluate.py`) — score each one (accuracy, precision, recall, F1, ROC-AUC), generate a confusion matrix and classification report for each, and compare them side by side.
7. **Tune the best one** (`tune.py`) — Random Forest won, so we run `GridSearchCV`, which automatically tries 81 different combinations of settings (like "how many trees" and "how deep can each tree grow") and keeps the best-scoring one.
8. **Save everything** — the final tuned model (`models/best_model.joblib`), all charts (`reports/figures/`), and all result tables (`reports/results/`), plus a full text log of the whole run (`reports/logs/pipeline.log`).

---

## 5. Key design decisions, explained as "why this, not that"

**Why `class_weight='balanced'` instead of just training normally?**
With only 3.4% failures, a model trained normally would learn "just always guess no-failure" and be right 96.6% of the time while never catching a single real failure. `class_weight='balanced'` tells the model "treat mistakes on the rare failure cases as more costly," forcing it to actually try to catch them.

**Why not use SMOTE (a technique that generates synthetic extra failure examples) instead?**
SMOTE is a legitimate technique, but it adds real complexity (it invents synthetic data points, which can occasionally create unrealistic examples) and was explicitly scoped out for this phase. We use the simpler `class_weight='balanced'` approach and *document* SMOTE as a future improvement rather than implementing it — a deliberate scope decision, not an oversight.

**Why tune only Random Forest, not all 3 models?**
Random Forest was clearly the best baseline (98.2% accuracy vs. 82–93% for the others). Tuning is expensive (GridSearchCV here tries 81 combinations × 5 checks each = 405 training runs, taking several minutes). Spending that time tuning the two weaker models wouldn't change which model we ultimately use, so it's time better spent elsewhere. This is stated explicitly in the README as a scoping decision.

**Why is preprocessing (scaling numbers, encoding categories) built *into* each model's pipeline instead of done once upfront?**
If you scaled the *entire* dataset before splitting into train/test, information from the test set (its average, its range) would "leak" into how the training data gets scaled — a subtler version of the same leakage problem from Section 2. Building the scaler into the `Pipeline` means it's re-fit *only* on the training data every time, and the exact same transformation is then applied to the test data without letting the test data influence it.

**Why `StandardScaler` for numbers but `OneHotEncoder` for `Type`?**
`Air_Temperature`, `Torque`, etc. are numbers where "bigger" and "smaller" are meaningful — scaling puts them all on a comparable range so no single feature dominates just because its raw numbers happen to be bigger. `Type` (L/M/H) has no numeric meaning — M isn't "between" L and H mathematically — so it's converted into separate yes/no columns instead (one-hot encoding).

**Why `drop='first'` in the one-hot encoder?**
With 3 categories (L/M/H), you only need 2 yes/no columns to fully describe them (if it's not L and not M, it must be H). Keeping a 3rd redundant column doesn't add information and can confuse some models — so we drop one.

**Why `RANDOM_STATE = 42` everywhere, defined once?**
Lots of steps here involve randomness — which rows go into training vs. testing, how a Random Forest picks its trees, etc. Fixing the seed means running the project twice gives you the *exact same numbers* both times — essential for a reproducible academic result. It's defined once in `config.py` and imported everywhere so it's guaranteed to be the same seed everywhere, rather than accidentally using different random behavior in different files.

**Why does `run_pipeline.py` exist separately from the individual scripts, if they already do everything?**
So there's one single command (`python -m src.run_pipeline`) for a full clean run — useful when you just want the final result — while the individual scripts remain useful for inspecting or debugging one stage without waiting for the whole thing.

**Why `python -m src.run_pipeline` instead of `python src/run_pipeline.py`?**
The `-m` form tells Python "run this as part of the `src` package," which makes the `from src.config import ...`-style imports inside the file work correctly regardless of which folder you happened to run the command from, and behaves identically on Windows and Mac/Linux. Running the file directly can break those imports depending on your current folder.

**Why `pathlib.Path` instead of writing paths like `"data/raw/" + filename`?**
Windows uses `\` and Mac/Linux use `/` in file paths. `pathlib.Path` handles that difference automatically, so the exact same code works on both without anyone having to remember which slash to use.

**Why does every plotting function both save a PNG *and* return the figure object?**
The pipeline scripts just need the PNG saved to disk. The notebook needs to *display* the chart inline for someone reading it. Rather than writing the plotting logic twice, each function does both — save the file, and also hand back the chart object so the notebook can show it. One function, two use cases.

**Why joblib instead of Python's built-in `pickle` to save the model?**
`joblib` is optimized for saving objects that contain large numeric arrays (which is exactly what a trained Random Forest is internally) — faster and produces smaller files than plain `pickle` for this kind of object.

**Why log to both the console *and* a file, instead of just printing?**
Printing to the console is good for watching progress live. Saving to `reports/logs/pipeline.log` means you have permanent proof of exactly what happened on a specific run (useful for showing a professor "here's the actual output," or debugging something that happened yesterday). `src/logger.py` sets up both at once so every other file gets both automatically just by asking for a logger.

**Why `scoring='f1'` for GridSearchCV instead of accuracy?**
Accuracy is misleading here — a model that never predicts "failure" still gets 96.6% accuracy. F1 balances precision (don't cry wolf) and recall (don't miss real failures), which is a much more honest way to pick the best settings for this specific imbalanced problem.

---

## 6. How Random Forest actually works (simple version)

A single Decision Tree asks a series of yes/no questions ("Is torque above 45?" → "Is tool wear above 200?" → ...) to reach a guess. One tree can overfit — it might learn quirks specific to the exact training data that don't generalize.

A **Random Forest** builds many different decision trees (200, in this project), each one trained on a slightly different random slice of the data and allowed to look at slightly different features at each question. Then it takes a majority vote across all of them. This "wisdom of crowds" effect is why it beat a single Decision Tree and Logistic Regression here — it's more resistant to the training data's quirks.

---

## 7. What `predict.py` does differently from everything else

Every other script measures how good the model is, using data where we already know the right answer. `predict.py` is the only file that uses the finished, saved model (`models/best_model.joblib`) to answer a question we *don't* already know the answer to — you type in a new machine's readings, and it tells you what it thinks will happen. This is the "put it to actual use" step, versus everything before it being "build and check it works."

---

## 8. What the notebook is for, versus the scripts

The scripts (`src/`) are built to be run by a computer, repeatedly, unattended. The notebook (`notebooks/exploratory_analysis.ipynb`) is built to be *read by a person* — it has explanations in plain English between each chart, walking through what the data looks like and why. It calls the exact same functions as the scripts (no duplicated logic — see Section 3), it's just packaged for a human audience instead of automation. This is the file to open when presenting to your professor.

---

## 9. The known limitation, and why it's acceptable

The final model catches about 71% of real failures (recall) — meaning about 29% of real failures slip through undetected. This is a direct, unavoidable consequence of only having 339 failure examples to learn from out of 10,000. `class_weight='balanced'` helps, but doesn't fully solve it. Techniques like SMOTE, adjusting the decision threshold, or cost-sensitive learning could improve this further — they're documented as future work rather than implemented, because doing them properly requires careful validation to avoid making the model *overconfident* rather than actually better, and that was out of scope for this phase.

# Data

Put one of the following here (nothing is downloaded automatically):

1. **Kaggle "IMDB Dataset of 50K Movie Reviews"** (the dataset used in the paper)
   https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
   -> save as `data/IMDB Dataset.csv` (columns: `review`, `sentiment`).
2. **Stanford aclImdb v1** (official 25k/25k train/test split, filenames carry the 1-10 rating)
   https://ai.stanford.edu/~amaas/data/sentiment/  -> extract to `data/aclImdb/`.
3. **Any CSV** with a text column (`review`/`text`/`sentence`) and a label column
   (`sentiment`/`label`: positive/negative/neutral or 0/1) or a numeric `rating` column.

No data? `python -m msa.train --demo` generates a small synthetic set (pipeline smoke test only; numbers are NOT meaningful).

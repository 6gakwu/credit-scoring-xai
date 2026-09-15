# Dataset

This project uses the **Home Credit Default Risk** dataset from Kaggle:

https://www.kaggle.com/c/home-credit-default-risk

The dataset is not included in this repository, in line with Kaggle's competition data terms. To reproduce the results:

1. Create a free Kaggle account if you don't already have one.
2. Go to the [competition data page](https://www.kaggle.com/c/home-credit-default-risk/data) and accept the competition rules (required before download is enabled).
3. Download the following files:
   - `application_train.csv`
   - `application_test.csv`
   - `bureau.csv`
   - `previous_application.csv`
   - `installments_payments.csv`
4. Place all five files in `data/raw/` (create this folder if it doesn't exist).

Only these five tables are used in this project. The dataset's other supporting tables (`bureau_balance.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`) are not used, see Chapter 3 of the full report for why.

Once the raw files are in place, follow the reproduction steps in the main [README](../README.md#reproducing-the-results).

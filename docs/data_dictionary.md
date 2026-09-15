# Data Dictionary

Every feature present in the final modelling dataset (`train_selected.csv` / `test_selected.csv`), organised by source table, with a description and derivation for each.

## Application-Level Features

Covers every applicant-level column present in the final merged dataset. TARGET is present only in the training data.

| Feature | Description | Derivation |
|---|---|---|
| SK_ID_CURR | Unique applicant identifier. |  |
| TARGET | 1 = applicant had payment difficulty/default. 0 = no payment difficulty. Present only in train_merged.csv. |  |
| NAME_CONTRACT_TYPE | Type of loan contract requested. | 0 = Cash loans, 1 = Revolving loans. |
| FLAG_OWN_CAR | Whether the applicant owns a car. | 0 = No, 1 = Yes. |
| FLAG_OWN_REALTY | Whether the applicant owns a house or flat. | 0 = No, 1 = Yes. |
| CNT_CHILDREN | Number of children the applicant has. |  |
| AMT_INCOME_TOTAL | Total income of the applicant. |  |
| AMT_CREDIT | Total credit amount requested or granted for the current loan. |  |
| AMT_ANNUITY | Regular repayment amount for the current loan. | 12 originally-missing rows filled with the training-set median. |
| NAME_TYPE_SUITE | Who accompanied the applicant when applying. | Smoothed target encoding: category replaced with its average TARGET rate, weighted toward the overall training rate by category size. Missing values grouped as “Unknown” before encoding. |
| NAME_INCOME_TYPE | Applicant’s income type (e.g. Working, Pensioner, Commercial associate). | Smoothed target encoding. |
| NAME_EDUCATION_TYPE | Highest education level achieved. | Ordinal: 0 = Lower secondary, 1 = Secondary / secondary special, 2 = Incomplete higher, 3 = Higher education, 4 = Academic degree. |
| NAME_FAMILY_STATUS | Family or marital status of the applicant. | Smoothed target encoding. |
| NAME_HOUSING_TYPE | Applicant’s housing situation. | Smoothed target encoding. |
| REGION_POPULATION_RELATIVE | Normalized population level of the applicant’s region. Higher = more populated. |  |
| OWN_CAR_AGE | Age of the applicant’s car, if owned. | Left blank (NaN) for applicants with no car (FLAG_OWN_CAR = 0) and for a small number of car owners with a genuinely unrecorded age. |
| FLAG_EMP_PHONE | Whether the applicant provided an employment phone number. |  |
| FLAG_PHONE | Whether the applicant provided a phone number. |  |
| FLAG_EMAIL | Whether the applicant provided an email address. |  |
| OCCUPATION_TYPE | Applicant’s occupation category. | Smoothed target encoding. Originally-missing values (“Unknown”) encoded as their own category. |
| CNT_FAM_MEMBERS | Number of family members in the applicant’s household. | 2 originally-missing rows filled with the training-set median. |
| REGION_RATING_CLIENT_W_CITY | Regional risk rating adjusted with city-level information (1 = Low Risk, 3 = High Risk). |  |
| HOUR_APPR_PROCESS_START | Hour of the day the applicant submitted the loan application. |  |
| REG_REGION_NOT_WORK_REGION | Whether the applicant’s registered region differs from their work region. |  |
| LIVE_REGION_NOT_WORK_REGION | Whether the applicant’s living region differs from their work region. |  |
| REG_CITY_NOT_WORK_CITY | Whether the applicant’s registered city differs from their work city. |  |
| LIVE_CITY_NOT_WORK_CITY | Whether the applicant’s living city differs from their work city. |  |
| ORGANIZATION_TYPE | Type of organization where the applicant works. | Smoothed target encoding. |
| EXT_SOURCE_1 | Anonymized normalized score from an external data source. | Genuinely-missing values (56.4%) imputed with a linear regression model trained on applicant age, income, credit amount, employment years, EXT_SOURCE_2, and EXT_SOURCE_3, fitted on training data only. |
| EXT_SOURCE_1_MISSING | Whether EXT_SOURCE_1 was originally missing before imputation. | 1 if missing prior to imputation, else 0. |
| EXT_SOURCE_2 | Anonymized normalized score from an external data source. | Genuinely-missing values (0.2%) filled with the training-set median. |
| EXT_SOURCE_2_MISSING | Whether EXT_SOURCE_2 was originally missing before imputation. | 1 if missing prior to imputation, else 0. |
| EXT_SOURCE_3 | Anonymized normalized score from an external data source. | Genuinely-missing values (19.8%) imputed with a linear regression model trained on applicant age, income, credit amount, employment years, and EXT_SOURCE_2, fitted on training data only. |
| EXT_SOURCE_3_MISSING | Whether EXT_SOURCE_3 was originally missing before imputation. | 1 if missing prior to imputation, else 0. |
| OBS_30_CNT_SOCIAL_CIRCLE | Number of observations in the applicant’s social circle with 30 days past due. | Originally-missing rows (0.33%) filled with the training-set median. |
| DEF_30_CNT_SOCIAL_CIRCLE | Number of observations in the applicant’s social circle that defaulted at 30 days past due. | Originally-missing rows filled with the training-set median. |
| DEF_60_CNT_SOCIAL_CIRCLE | Number of observations in the applicant’s social circle that defaulted at 60 days past due. | Originally-missing rows filled with the training-set median. |
| SOCIAL_CIRCLE_MISSING | Whether the applicant’s social circle counts were originally missing. | 1 if all four social circle columns were missing, else 0. |
| AMT_REQ_CREDIT_BUREAU_HOUR | Number of credit bureau enquiries about the applicant in the hour before application. | Originally-missing rows (13.5%, applicants with no bureau history) filled with 0. |
| AMT_REQ_CREDIT_BUREAU_DAY | Number of credit bureau enquiries in the day before application. | Originally-missing rows filled with 0. |
| AMT_REQ_CREDIT_BUREAU_WEEK | Number of credit bureau enquiries in the week before application. | Originally-missing rows filled with 0. |
| AMT_REQ_CREDIT_BUREAU_MON | Number of credit bureau enquiries in the month before application. | Originally-missing rows filled with 0. |
| AMT_REQ_CREDIT_BUREAU_QRT | Number of credit bureau enquiries in the quarter before application. | Originally-missing rows filled with 0. |
| AMT_REQ_CREDIT_BUREAU_YEAR | Number of credit bureau enquiries in the year before application. | Originally-missing rows filled with 0. |
| AGE_YEARS | Applicant’s age in years. | AGE_YEARS = (-DAYS_BIRTH) / 365. |
| EMPLOYMENT_YEARS | Number of years since the applicant started their current employment. | EMPLOYMENT_YEARS = (-DAYS_EMPLOYED) / 365, set to 0 for Pensioner/Unemployed applicants (NAME_INCOME_TYPE), for whom a current employment duration does not apply. |
| REGISTRATION_YEARS | Number of years since the applicant last changed registration. | REGISTRATION_YEARS = (-DAYS_REGISTRATION) / 365. |
| YEARS_SINCE_ID_UPDATE | Number of years since the applicant last changed their identity document. | YEARS_SINCE_ID_UPDATE = (-DAYS_ID_PUBLISH) / 365. |
| YEARS_SINCE_PHONE_CHANGE | Number of years since the applicant last changed phone number. | YEARS_SINCE_PHONE_CHANGE = (-DAYS_LAST_PHONE_CHANGE) / 365. 1 originally-missing row filled with the training-set median. |
| CREDIT_TO_INCOME_RATIO | Loan amount relative to applicant income. | AMT_CREDIT / AMT_INCOME_TOTAL. |
| ANNUITY_TO_INCOME_RATIO | Regular repayment burden relative to applicant income. | AMT_ANNUITY / AMT_INCOME_TOTAL. 12 originally-missing rows (inherited from AMT_ANNUITY) filled with the training-set median. |
| CREDIT_GOODS_RATIO | Loan amount relative to the price of the goods being financed. Replaces AMT_GOODS_PRICE (0.987 correlated with AMT_CREDIT). | AMT_CREDIT / AMT_GOODS_PRICE. Missing ratios filled with the training-set median ratio. |
| CODE_GENDER_M | Whether the applicant’s recorded gender is Male. | One-hot indicator, fit on training categories. |

## Bureau History (Aggregated)

Summarises each applicant's external credit bureau history from bureau.csv, aggregated to one row per applicant.

| Feature | Description | Derivation |
|---|---|---|
| TOTAL_PREVIOUS_LOANS | Total number of previous bureau credit records for the applicant. | Count of SK_ID_BUREAU per SK_ID_CURR. |
| ACTIVE_PREVIOUS_LOANS | Number of previous bureau credits still active. | Count of CREDIT_ACTIVE = “Active” per SK_ID_CURR. |
| TOTAL_BUREAU_DEBT | Total outstanding debt across the applicant’s previous bureau credits. | Sum of AMT_CREDIT_SUM_DEBT per SK_ID_CURR. |
| TOTAL_OVERDUE_DEBT | Total overdue debt across the applicant’s previous bureau credits. | Sum of AMT_CREDIT_SUM_OVERDUE per SK_ID_CURR. |
| MAX_DAYS_OVERDUE | Worst recorded overdue delay in previous bureau credits. | Max of CREDIT_DAY_OVERDUE per SK_ID_CURR. |
| MOST_RECENT_PREVIOUS_CREDIT_YEARS | How recently the applicant took a previous bureau credit. | Minimum of (-DAYS_CREDIT / 365) per SK_ID_CURR. |
| COUNT_CREDIT_CARDS | Number of previous credit card records. | Count of CREDIT_TYPE = “Credit card” per SK_ID_CURR. |

## Previous Home Credit Applications (Aggregated)

Summarises the applicant's past Home Credit application history from previous_application.csv, aggregated to one row per applicant.

| Feature | Description | Derivation |
|---|---|---|
| TOTAL_PREVIOUS_APPLICATIONS | Total number of previous Home Credit applications made by the applicant. | Count of SK_ID_PREV per SK_ID_CURR. |
| APPROVED_PREVIOUS_APPLICATIONS | Number of previous applications that were approved. | Count of NAME_CONTRACT_STATUS = “Approved” per SK_ID_CURR. |
| REFUSED_PREVIOUS_APPLICATIONS | Number of previous applications that were refused. | Count of NAME_CONTRACT_STATUS = “Refused” per SK_ID_CURR. |
| PREVIOUS_APPROVAL_RATE | Proportion of previous applications that were approved. | APPROVED_PREVIOUS_APPLICATIONS / TOTAL_PREVIOUS_APPLICATIONS. |
| PREVIOUS_REFUSAL_RATE | Proportion of previous applications that were refused. | REFUSED_PREVIOUS_APPLICATIONS / TOTAL_PREVIOUS_APPLICATIONS. |
| AVG_PREVIOUS_APPLICATION_AMOUNT | Average amount the applicant previously applied for. | Mean of AMT_APPLICATION per SK_ID_CURR. |
| MAX_PREVIOUS_CREDIT_AMOUNT | Largest previous credit amount granted to the applicant. | Max of AMT_CREDIT per SK_ID_CURR. |
| MOST_RECENT_PREVIOUS_APPLICATION_YEARS | How recently the applicant had a previous Home Credit application. | Minimum of (-DAYS_DECISION / 365) per SK_ID_CURR. |

## Instalment Repayment History (Aggregated)

Summarises actual repayment behaviour from installments_payments.csv, aggregated to one row per applicant.

| Feature | Description | Derivation |
|---|---|---|
| TOTAL_INSTALLMENT_RECORDS | Total number of installment/payment records available for the applicant. | Count of SK_ID_PREV per SK_ID_CURR. |
| LATE_PAYMENT_COUNT | Number of installment payments made after the scheduled due date. | Count of records where (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) > 0. |
| LATE_PAYMENT_RATE | Proportion of the applicant’s installment payments that were late. | LATE_PAYMENT_COUNT / TOTAL_INSTALLMENT_RECORDS. |
| AVG_LATE_PAYMENT_DAYS | Average late-payment delay in days. Early/on-time payments count as 0. | Mean of max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0) per SK_ID_CURR. |
| MAX_LATE_PAYMENT_DAYS | Worst/longest late-payment delay recorded. Early/on-time payments count as 0. | Max of max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0) per SK_ID_CURR. |


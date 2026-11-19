# Dataset folder: sample_data

Source repository: `JayanGupta/AutoData`

This folder contains datasets for data-analysis and machine-learning experiments.

| Dataset | Rows | Columns | Fields | Purpose | Suitable analysis |
| --- | ---: | ---: | --- | --- | --- |
| `customer_churn.csv` | 1501 | 11 | `customer_id`, `tenure_months`, `monthly_charges`, `total_charges`, `plan`, `contract`, `payment_method`, `senior_citizen`, `support_calls`, `avg_call_minutes`, `churn` | Business and retail analysis | schema profiling, missing-value and duplicate checks, grouped summaries, customer segmentation |
| `sales_data.csv` | 1201 | 11 | `order_id`, `order_date`, `product`, `category`, `region`, `channel`, `units`, `unit_price`, `revenue`, `customer_rating`, `returned` | Business and retail analysis | schema profiling, missing-value and duplicate checks, grouped summaries, price or revenue comparisons, trend and ranking analysis, rating or sentiment analysis, time-series analysis, customer segmentation |
| `web_traffic.csv` | 365 | 6 | `date`, `visits`, `sessions`, `signups`, `bounce_rate`, `avg_session_minutes` | Exploratory data analysis and machine-learning preparation | schema profiling, missing-value and duplicate checks, grouped summaries, time-series analysis |

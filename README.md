# MSc BAM Master Thesis

**Sanjana Aneja | 2025-2026**

```
The copyright of the Master Thesis rests with the author. The author is responsible for
its contents. RSM is only responsible for the educational coaching and cannot be held
liable for the content.

The code in this repository (excluding FlightDelayPrediction/, which is the original
work of Tan et al. (2025)) was written by Sanjana Aneja.
```

This repository contains the code used to produce all results in the thesis
*"Network-Spilled Propagation
across U.S. Airport Hub
Categories: Varying flight
delay predictions due to Hub
Specific Dynamics"*.

The thesis investigates whether the ANSP delay score, originally developed for a limited scope of only 
large hubs by Tan et al. (2025), also improves delay prediction when considering a unified network structure of 60 airports that better represents the reality of delay propogation that exists in U.S domestic airspace. This thesis tests whether ANSP generalises to all hubs and whether hub-specific parameter tuning adds further value.

---

## Data

The raw data files are not included in this repository because of their size.
Three sources are required to run the pipeline from scratch.

| File | Source | Used in thesis |
|------|--------|----------------|
| `BTS_ONTIME_REPORTING_June_2023.csv` | [BTS On-Time Reporting](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD) | Section 3.1 |
| `BTS_ONTIME_REPORTING_July_2023.csv` | [BTS On-Time Reporting](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD) | Section 3.1 |
| `FAA_selected_airports_sample.csv` | [FAA Hub Classifications](https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger) | Section 3.2 |
| `weather_data/*.csv` | IEM ASOS network via `iem_weather_day.py` | Section 3.3 |

Place the BTS and FAA files in the root directory. Weather files are fetched by
`iem_weather_day.py` and saved to `weather_data/`.

---

## Repository Structure

```
master_thesis/
|
|-- run_pipeline.py                   Entry point. Runs all pipeline stages.
|
|-- Stage 1: create_thesis_input_files.py
|-- Stage 2: calculate_traffic_metrics.py
|-- Stage 3: calculate_turnaround_metrics.py
|-- Stage 4: build_unified_60.py
|-- Stage 5: thesis_main.py
|-- Stage 6: data_and_methods_outputs.py
|-- Stage 7: thesis_empirical_validation.py
|
|-- rq2_parameter_search.py           RQ2 parameter grid search
|
|-- helper.py                         Shared utility functions
|-- delay_score.py                    ANSP delay score computation
|-- network_feature.py                Network centrality features
|-- airport_network_mapping.py        Airport to IEM weather station mapping
|-- iem_weather_day.py                Downloads weather data from IEM API
|-- empirical_validation_out_of_time.py  Out-of-time validation logic
|
|-- thesis_outputs/                   All outputs produced by the pipeline
|   |-- delay_score/                  ANSP scores per month (global parameters)
|   |-- network_feature/              PageRank centrality per month
|   |-- figures/                      All thesis figures
|   |-- results/                      Model evaluation results (Tables 4 and 5)
|   |   |-- empirical_results.csv     5-fold cross-validation results
|   |   |-- empirical_results_oot.csv Out-of-time validation results
|   |-- tables/                       Summary statistics and data tables
|   |-- rq2/                          RQ2 parameter search outputs
|
|-- weather_data/                     Raw weather CSVs per state
|-- FlightDelayPrediction/            Original code from Tan et al. (2025)
```

---

## Pipeline Stages

Run the full pipeline with:

```bash
python run_pipeline.py
```

Or run specific stages:

```bash
python run_pipeline.py --stages 5-7   # run stages 5, 6, and 7
python run_pipeline.py --from 4       # resume from stage 4
python run_pipeline.py --list         # print all stages and exit
```

| Stage | Script | What it does | Thesis section |
|-------|--------|--------------|----------------|
| 1 | `create_thesis_input_files.py` | Filters BTS data into Large, Medium, and Small hub files. Merges weather and computes time columns. | Section 3, Section 5.4 |
| 2 | `calculate_traffic_metrics.py` | Computes 6 traffic congestion features (arrivals, departures, short turnaround) across the full BTS dataset. | Appendix B |
| 3 | `calculate_turnaround_metrics.py` | Computes turnaround time for flights where the inbound leg is also in the BTS dataset. | Appendix B |
| 4 | `build_unified_60.py` | Merges the three hub files into one unified 60-airport dataset (`input_data_unified_60.csv`). | Section 5, Table 1 |
| 5 | `thesis_main.py` | Computes ANSP delay scores (alpha=0.85, beta=2.3, gamma=0.9) and PageRank centrality for each month. | Section 4.4, Section 5.3 |
| 6 | `data_and_methods_outputs.py` | Generates Figure 6 (delay distribution by hub) and Figure 7 (daily delay timeseries). | Section 5.4 |
| 7 | `thesis_empirical_validation.py` | Trains XGBoost, Random Forest, Bagged LR, and ANN models. Runs 5-fold CV and saves results. | Section 6.1, Tables 4-6, Figures 8-9 |

After stage 7, run out-of-time validation and RQ2 separately:

```bash
# Out-of-time validation (Table 5)
python empirical_validation_out_of_time.py

# RQ2 parameter grid search (Table 7, Figure 10, Figure 11)
python rq2_parameter_search.py
```

---

## Key Output Files

| File | Thesis location |
|------|-----------------|
| `thesis_outputs/results/empirical_results.csv` | Table 4 (5-fold CV, all models and hubs) |
| `thesis_outputs/results/empirical_results_oot.csv` | Table 5 (out-of-time validation) |
| `thesis_outputs/tables/ch5_ansp_vs_zscore.csv` | Table 6 (ANSP lift vs z-score lift) |
| `thesis_outputs/tables/rq2_best_per_hub.csv` | Table 7 (best parameters per hub) |
| `thesis_outputs/tables/rq2_default_vs_best.csv` | Table 8 (baseline vs default vs tuned ANSP) |
| `thesis_outputs/figures/delay_distribution_by_hub.pdf` | Figure 6 |
| `thesis_outputs/figures/daily_delay_timeseries.pdf` | Figure 7 |
| `thesis_outputs/figures/results_rq1_auc_by_lag.png` | Figure 8 (AUC by lookback window) |
| `thesis_outputs/figures/results_rq1_shap_importance.png` | Figure 9 (SHAP feature importance) |
| `thesis_outputs/figures/fig_rq2_heatmap_*.png` | Figure 10 (parameter sensitivity heatmaps) |
| `thesis_outputs/figures/fig_rq2_auc_comparison.png` | Figure 11 (baseline vs ANSP comparison) |
| `thesis_outputs/delay_score/` | ANSP scores used as model features (Section 4.4) |
| `thesis_outputs/network_feature/` | PageRank centrality features (Section 4.4) |

---

## Shared Modules

| File | Purpose |
|------|---------|
| `helper.py` | Date range generation, lambda calculation, data preprocessing |
| `delay_score.py` | ANSP delay score: builds the delay vector, weight matrix W, and runs Personalised PageRank propagation |
| `network_feature.py` | Builds the frequency matrix and computes in-degree, out-degree, betweenness, and PageRank centrality |
| `airport_network_mapping.py` | Maps each of the 60 airports to its IEM ASOS weather station network |
| `iem_weather_day.py` | Calls the IEM API to download hourly ASOS weather data by state |
| `empirical_validation_out_of_time.py` | Standalone out-of-time validation: trains on June 2023, tests on July 2023 |

---

## FlightDelayPrediction

The `FlightDelayPrediction/` folder contains the original code from Tan et al. (2025),
which this thesis extends. It is included for reference and comparison. It has its own
`README.md` with setup instructions for that codebase.

---

## Reproducibility Note

The ML models in stage 7 and the out-of-time validation do not use a fixed random
seed. Results will vary slightly across runs due to stochastic model training. The
numbers published in the thesis correspond to the files in `thesis_outputs/results/`.

---

## Environment

Python 3.12. Install dependencies with:

```bash
python -m venv .venv_win
.venv_win\Scripts\activate
pip install numpy pandas networkx matplotlib scipy sympy scikit-learn seaborn xgboost shap tqdm requests tensorflow
```

---

## AI Assistance Disclosure

In accordance with the RSM AI Guidance for Students.

```
AI tools were used for code formatting, parsing, and summarising repository structure.
All reasoning, implementation decisions, and verification were performed by me.
```

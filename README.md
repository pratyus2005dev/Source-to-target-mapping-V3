# Source-to-target-mapping-V3
Source-to-Target Mapping V3

Overview

Source-to-Target Mapping V3 is an advanced framework built to automate data migration, mapping, and ETL (Extract-Transform-Load) generation.

Data migration is one of the most challenging parts of system modernization — legacy databases often have hundreds of tables and columns that need to be mapped, transformed, and loaded into new systems. This project automates much of that work by:

Profiling data to understand schema, statistics, and semantics.

Generating mapping suggestions using machine learning models trained on labeled source–target pairs.

Automating ETL scripts for seamless migration.

Providing visualization and reporting for tracking mapping accuracy and model performance.

It is particularly designed for insurance and financial domain migrations, but is flexible enough to adapt to other industries where structured relational data migration is required.

Motivation

Traditional migration projects often rely on manual mapping — a slow, error-prone, and expensive process. With large legacy systems, it can take months of work just to define mappings before migration begins.

This project aims to:

Reduce manual effort by suggesting mappings automatically.

Improve mapping accuracy using ensemble ML models (RandomForest, HistGradientBoosting, CatBoost).

Provide extensibility so organizations can adapt the pipeline to new schemas or domains.

Enable data-driven migration decisions through profiling, visualization, and evaluation.

Key Features

🔎 Automated Data Profiling: Extracts schema metadata, value distributions, and similarity metrics.

🤖 ML-based Mapping Suggestion Engine: Scores and ranks potential source–target column matches.

⚡ ETL Automation: Generates migration-ready specifications and table creation scripts.

📊 Dashboard & Reporting: Visualizes mapping quality, profiling stats, and model performance.

⚙️ Configurable Pipeline: YAML/JSON-driven configuration for source/target pairs, training, prediction, and synonyms.

🔄 Extensibility: Easy to add new features, models, or mapping heuristics.

Project Structure
Source-to-target-mapping-V3/
│
├── config.py                # Config loader
├── configs/                 # Configurations
│   ├── config.yaml          # Main pipeline config
│   └── synonyms.json        # Ground truth / synonym mappings
│
├── dashboard.py             # Dashboard utilities
├── DDL.sql                  # Example DB2-style schema definitions
├── generate_mappings.py     # Entry point for mapping suggestion
├── generate_tables.py       # Table creation & ETL automation
├── merge.py                 # Transformation & merging utilities
├── synonyms.py              # Synonym handler
│
├── catboost_info/           # Model training logs & metrics
├── data/                    # Data directory
│   ├── source/              # Legacy source CSVs
│   └── target/              # Target system CSVs
│
├── models/                  # Saved ML model artifacts
├── outputs/                 # Mapping results & reports
│
├── src/                     # Core logic
│   ├── features.py          # Feature engineering
│   ├── train.py             # Training pipeline
│   ├── predict.py           # Prediction pipeline
│   ├── evaluate.py          # Evaluation utilities
│   ├── profiler.py          # Data profiling
│   └── utils/               # Helpers (I/O, text similarity, etc.)
│
└── requirements.txt         # Python dependencies

Pipeline Workflow

The pipeline has five major stages:

Data Profiling & Feature Engineering

Analyze schema and data values.

Generate features (data types, statistical overlaps, string similarity, frequency analysis).

Mapping Suggestion

For each source column, ML models score possible target matches.

Ranked suggestions are saved to CSV/JSON.

Model Training

Supports RandomForest, HistGradientBoosting, and CatBoost.

Negative sampling, train/test split, and evaluation metrics (F1, AUC, Average Precision).

ETL Automation

Based on mappings, generates SQL ETL specifications and table creation scripts.

Dashboard & Reporting

Summarizes mapping performance, profiling results, and model metrics.

Configuration

All settings are controlled via configs/config.yaml.

Key sections include:

Source/Target Inputs

source:
  root: "data/source/"
  files: ["LEG_POLICY.csv", "LEG_CUSTOMER.csv"]

target:
  root: "data/target/"
  files: ["LP_POLICY.csv", "LP_CUSTOMER.csv"]


Table Pairs

table_pairs:
  - source: "LEG_POLICY"
    target: "LP_POLICY"
  - source: "LEG_CUSTOMER"
    target: "LP_CUSTOMER"


Training

train:
  model_output: "models/matcher.pkl"
  negative_ratio: 2
  test_size: 0.2
  random_seed: 42


Prediction

predict:
  model_input: "models/matcher.pkl"
  top_k: 3
  threshold: 0.6
  output_csv: "outputs/mapping_suggestions.csv"


Synonyms

synonyms_path: "configs/synonyms.json"

Data Preparation

Place legacy source data in data/source/.

Place target system data in data/target/.

Ensure filenames match those in config.yaml.

Update DDL.sql with schema definitions (if needed).

(Optional) Add synonyms.json for known mappings.

Usage
1. Install Dependencies
pip install -r requirements.txt

2. Generate Tables & ETL Scripts
python generate_tables.py

3. Generate Mapping Suggestions
python generate_mappings.py

4. Train Models
python src/train.py --config configs/config.yaml

5. Predict/Score Mappings
python src/predict.py --config configs/config.yaml

6. Evaluate Results
python src/evaluate.py

7. View Outputs

Check outputs/ for CSV/JSON mappings, ETL specs, and reports.

Outputs

outputs/mapping_suggestions.csv → Ranked mapping suggestions

outputs/mapping_suggestions.json → JSON equivalent

outputs/etl_spec.csv → Migration-ready ETL specifications

catboost_info/ → Training logs & plots

models/matcher.pkl → Trained model artifact

Requirements

Python 3.8+

Libraries: pandas, numpy, scikit-learn, catboost, PyYAML, joblib, etc.

Install via requirements.txt

Best Practices

Keep training and prediction data separate.

Use synonyms.json for better accuracy with domain-specific mappings.

Regularly retrain models as schemas evolve.

Validate mapping outputs with SMEs (Subject Matter Experts).

Extending the Project

Add new features in src/features.py.

Plug in new ML models in src/train.py.

Extend I/O logic for different DB engines in src/utils/io_utils.py.

Customize ETL generation for your organization’s migration strategy.

Limitations & Future Work

Currently supports structured CSV/DDL-based inputs; extending to direct DB connections is possible.

Ensemble model weights are fixed; future versions may include automated model selection/tuning.

Dashboard can be expanded to provide real-time monitoring for migration projects.

# Higgs Boson Classification using PySpark

The HIGGS dataset contains simulated particle collision events used to distinguish Higgs boson signals from background processes. The binary classification task, built with PySpark, predicts whether an event represents a Higgs signal based on 28 physical features extracted from detector measurements. Experiments like the Large Hadron Collider produce billions of collisions, and only a small fraction are Higgs signals, so manual analysis is not practical.

**Dashboards:** [Tableau Public](https://public.tableau.com/app/profile/kowshiabhinav.konda/vizzes)

## Dataset

* **Source:** UCI Machine Learning Repository – HIGGS Dataset
* **File:** `HIGGS.csv`
* **Scale:** 11,000,000 rows and 29 columns.
* **Target:** `label` (Binary classification: 1 = HIGGS signal, 0 = background).
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **SparkSession (Task 4 run):** Local mode using PySpark with distributed processing on CPU.

## Repository Structure
```
.
├── Notebooks/
│   ├── Task1 Notebook.ipynb   # Problem & dataset exploration (5 Vs of big data)
│   ├── Task2 Notebook.ipynb   # Data engineering & PySpark pipeline
│   ├── Task3 Notebook.ipynb   # Training four ML models
│   ├── Task4 Notebook.ipynb   # Distributed computing: partitioning & caching
│   ├── Task5 Notebook.ipynb   # Evaluation, stability testing, LIME
│   └── Task6 Notebook.ipynb   # Outputs for Tableau dashboards
└── README.md
```

## Execution Environment

* **Engine:** Apache Spark (PySpark)
* **Compute:** CPU only (Spark MLlib)
* **Programming Language:** Python
* **Development Environment:** Jupyter Notebook

## Sampling Strategy

The HIGGS dataset contains approximately 11 million observations. The data is randomly split into 80% training data and 20% testing data using a fixed random seed to ensure reproducibility. The testing dataset remains unseen during model training and is used exclusively for evaluating model performance.

## Approach

1. **Data engineering:** renamed columns (`label`, `c1` to `c28`), confirmed there were no missing values, then built a PySpark ML `Pipeline` with `VectorAssembler` and `StandardScaler`.
2. **Modelling:** Logistic Regression, Decision Tree, Random Forest and Gradient Boosted Trees.
3. **Distributed computing:** repartitioned to 8 partitions and cached the DataFrame in memory, then inspected jobs and stages in the Spark UI.
4. **Evaluation:** accuracy, precision, F1, AUC-ROC, confusion matrices, ROC/PR curves, a perturbation stability test, and LIME explanations.
5. **Visualisation:** Tableau dashboards for class distribution, model performance, prediction confidence, and scalability and cost.


| Model                  |        Accuracy |  Precision      |   F1 Score      |    ROC-AUC   |
| ---------------------- | --------------: | --------------: | --------------: | --------------: 
| Logistic Regression    | *(0.635175)*    | *(0.635862)*    | *(0.629832)*    | *(0.679043)* 
| Decision Tree          | *(0.704703)*    | *(0.704703)*    | *(0.704703)*    | *(0.669855)* 
| Random Forest          | *(0.697418)*    | *(0.697084)*    | *(0.696473)*    | *(0.769555)* 
| Gradient Boosted Trees | *(0.718196)*    | *(0.718026)*    | *(0.718085)*    | *(0.795335)* 

Gradient Boosted Trees achieved the highest overall classification performance across the evaluation metrics, followed closely by Random Forest. Logistic Regression provided a strong linear baseline, while Decision Tree offered greater interpretability with comparatively lower predictive performance.

## Output Artifacts

The notebooks export the data files used in the Tableau dashboards.

  - `model_metrics.csv`
  - `feature_importance.csv`
  - `prediction_confidence.csv`
  - `scalability_results.csv`
  - `cost_analysis.csv`

## Tech Stack
* Python
* Jupyter
* PySpark (MLlib)
* LIME
* Tableau Public 

# Credit Card Customer Segmentation (Unsupervised Learning)

This project performs exploratory data analysis and applies unsupervised machine learning techniques to segment credit card customers based on their usage behavior over the last six months. The goal is to create **actionable customer segments** that can be used by marketing, product, and risk teams to tailor strategy and improve key business metrics (revenue, retention, risk-adjusted return).

## Dataset
- **File:** `data/CC_GENERAL.csv`
- **Source:** Kaggle ([arjunbhasin2013/ccdata](https://www.kaggle.com/arjunbhasin2013/ccdata))
- **Description:** Contains 18 behavioral variables for ~9,000 active credit card holders, including balances, purchase amounts, transaction frequencies, cash advances, credit limits, payments, and tenure.

## Problem & Business Context
Credit card portfolios are typically large and heterogeneous: some customers revolve balances and drive interest income, some pay in full but generate interchange revenue, others are low-usage or at higher risk. Without segmentation, it is difficult to:
- Target marketing campaigns efficiently
- Identify high-value customers for retention and upsell
- Monitor segments with elevated credit risk

This project builds **unsupervised customer segments** from behavioral features so that a product or CRM team can:
- Design tailored campaigns per segment (offers, rewards, credit line management)
- Track performance KPIs by segment (revenue, churn, delinquency)
- Prioritize analytics / product work where it has the most impact

## Objectives
1. Exploratory Data Analysis (EDA) and data cleaning  
2. Feature engineering and scaling (standardization)  
3. Dimensionality reduction using Principal Component Analysis (PCA)  
4. Customer segmentation via clustering:
   - KMeans (elbow method + cluster quality metrics)
   - Agglomerative (hierarchical) clustering
5. Evaluation with clustering metrics:
   - Silhouette score
   - Calinski–Harabasz index
   - Davies–Bouldin index
6. Visualization, segment profiling, and business interpretation

## Requirements
- Python 3.6+
- Jupyter Notebook
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - scipy

## Setup
1. Clone this repository.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```

## Usage

### 1. Run the end-to-end pipeline (scripts / code)

This repository includes a small Python module that runs the full unsupervised workflow (data ingestion, cleaning, scaling, clustering, and evaluation) outside of the notebook.

From the project root:

```bash
python3 -m src.pipeline
```

This will:
- Load and clean `data/CC_GENERAL.csv`
- Fit KMeans and hierarchical clustering models
- Compute clustering metrics (silhouette, Calinski–Harabasz, Davies–Bouldin)
- Save basic artifacts under `models/`:
  - `clustering_metrics.txt`
  - `cluster_size_distribution.png`

### 2. Explore the analysis in the notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open and run `notebooks/credit_card_segmentation.ipynb`.  
   Execute cells sequentially to reproduce the analysis and visualizations. The notebook now includes:
   - Clear problem framing and process overview
   - EDA and data preparation steps
   - Clustering with KMeans and hierarchical methods
   - PCA-based visualization
   - A concluding section with model evaluation, insights, and business impact

## Project Structure
```
├── data/
│   └── CC_GENERAL.csv              # Original dataset
├── models/
│   ├── clustering_metrics.txt      # Saved metrics from src.pipeline (created after running)
│   └── cluster_size_distribution.png  # Cluster size plot (created after running)
├── notebooks/
│   └── credit_card_segmentation.ipynb  # Exploratory analysis & clustering
├── src/
│   ├── __init__.py
│   └── pipeline.py                 # End-to-end clustering pipeline (ingestion → features → models → metrics)
└── README.md                       # Project overview and instructions
```

## Evaluation Metrics & Model Selection

This is an **unsupervised** problem (no ground-truth labels), so we evaluate and compare clustering models using internal metrics:
- **Silhouette score:** how similar each point is to its own cluster vs. other clusters (closer to 1 is better).
- **Calinski–Harabasz index:** ratio of between-cluster dispersion to within-cluster dispersion (higher is better).
- **Davies–Bouldin index:** average similarity between each cluster and its most similar one (lower is better).

The notebook and `src/pipeline.py` use these metrics alongside the elbow method to justify the choice of **three clusters**, balancing separation and stability.

## Results & Insights

- **EDA:** Identified and handled missing values; examined feature distributions and correlations to understand credit usage patterns.
- **PCA:** Reduced dimensionality while retaining ~95% of variance to make cluster visualization more interpretable.
- **KMeans:** Elbow method and clustering metrics suggested 3 clusters as a good trade-off between simplicity and separation.
- **Agglomerative:** Hierarchical clustering produced qualitatively similar segments, helping validate the cluster structure.
- **Segment interpretation (illustrative):**
  - Cluster 0: Low-usage / low-value customers (low balances, low purchase frequency, small credit limits).
  - Cluster 1: Mass affluent / revolving customers (higher balances and purchase activity, moderate cash advances).
  - Cluster 2: High-value / transactors (high spend and higher limits, relatively strong payment behavior).

Exact segment characteristics should be read from the cluster centroids and plots in the notebook; the labels above reflect the dominant behavioral patterns.

## Insights & Business Impact

Examples of how a product or marketing team could use these segments:
- **Targeted offers:** design differentiated campaigns per segment (e.g., rewards upgrades for high spenders, activation campaigns for dormant users).
- **Risk management:** apply more conservative credit line increases or closer monitoring to segments with heavy cash-advance usage or weak payment behavior.
- **Customer lifecycle:** track churn and revenue by segment to prioritize retention activities where ROI is highest.

Over time, this segmentation could improve:
- **Revenue per active account** (through better cross-sell and engagement)
- **Retention** (through more relevant offers and communications)
- **Risk-adjusted returns** (through better alignment of credit exposure with customer behavior)

## Acknowledgements
- Dataset provided by Arjun Bhasin on Kaggle (https://www.kaggle.com/arjunbhasin2013/ccdata).

## License
This project uses a public dataset subject to Kaggle's terms. No additional license is specified for the analysis code.

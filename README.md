# Credit Card Customer Segmentation (Unsupervised Learning)

This project performs exploratory data analysis and applies unsupervised machine learning techniques to segment credit card customers based on their usage behavior over the last six months. The analysis is conducted in a Jupyter Notebook using Python.

## Dataset
- **File:** `CC GENERAL.csv`
- **Source:** Kaggle ([arjunbhasin2013/ccdata](https://www.kaggle.com/arjunbhasin2013/ccdata))
- **Description:** Contains 18 behavioral variables for ~9,000 active credit card holders, including balances, purchase amounts, transaction frequencies, cash advances, credit limits, payments, and tenure.

## Objectives
1. Exploratory Data Analysis (EDA) and data cleaning
2. Feature scaling and correlation analysis
3. Dimensionality reduction using Principal Component Analysis (PCA)
4. Customer segmentation via clustering:
   - KMeans (elbow method to select optimal clusters)
   - Agglomerative (hierarchical) clustering
5. Visualization and interpretation of customer segments

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
1. Ensure `CC GENERAL.csv` is in the project root.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open and run `Credit Card - Unsupervised.ipynb`. Execute cells sequentially to reproduce the analysis and visualizations.

## Project Structure
```
├── CC GENERAL.csv               # Original dataset
├── Credit Card - Unsupervised.ipynb  # Exploratory and clustering analysis
└── README.md                    # Project overview and instructions
```

## Results Summary
- **EDA:** Identified and handled missing values; examined feature distributions and correlations.
- **PCA:** Reduced dimensionality while retaining 95% of variance.
- **KMeans:** Elbow method suggested 3–4 clusters; final segmentation used 3 clusters.
- **Agglomerative:** Hierarchical clustering produced similar segments.
- **Interpretation:** Customers grouped into low-, mid-, and high-usage segments based on purchase behavior and credit limits.

## Acknowledgements
- Dataset provided by Arjun Bhasin on Kaggle (https://www.kaggle.com/arjunbhasin2013/ccdata).

## License
This project uses a public dataset subject to Kaggle's terms. No additional license is specified for the analysis code.
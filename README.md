# ProductPulse 📊
### Retail Sales Analysis Using Machine Learning

## What's this project about?

I built ProductPulse as my B.Tech project to understand how machine learning can be used to analyze retail sales.

Retailers sell thousands of products across different stores, but not every product keeps selling at the same rate. Some have steady demand, while others experience a drop in sales. I wanted to see if historical sales data could help identify products that might have low sales in the future.

For this project, I used Walmart's M5 retail sales dataset and built a machine learning model that estimates whether a product is likely to sell fewer than 20 units over the next 90 days. I also made a Streamlit dashboard to explore the data and view the predictions.

## What I worked on

- Explored historical sales data for products across multiple stores.
- Studied recent sales, active selling days, and price changes.
- Created features using sales from the previous 28 and 90 days.
- Trained and compared Logistic Regression and Random Forest models.
- Evaluated the models using precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Built a Streamlit dashboard to view sales trends, product risk levels, and model results.

## Dataset

I used the **M5 Forecasting Accuracy dataset** from Kaggle. It contains historical Walmart sales data, calendar information, and product prices.

Dataset: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

The dataset files are not included here because they take up a lot of space. You'll need to download them from Kaggle and place the required files inside `data/raw/`.

## Models I tried

**Logistic Regression**

I used this as my baseline model. It estimates the probability that a product will fall into the low-future-sales category and gives me something to compare the other model against.

**Random Forest**

I also trained a Random Forest classifier, which combines multiple decision trees to make predictions. I tested versions with and without price-related features to see whether adding price information helped.

In my recorded experiments, the Random Forest model with price features performed better than the Logistic Regression baseline.

## Results

The Random Forest model with price features achieved:

- **F1-score:** approximately 0.715
- **PR-AUC:** approximately 0.790

These are average results from the recorded model comparisons. The results suggest that recent sales activity is useful for identifying products that may have low future sales, while price features provide a smaller improvement.

One important thing to note: the model predicts low future sales based on a threshold. It does not tell us whether a product has actually been discontinued.

## The dashboard

I built the dashboard using Streamlit so the results could be explored without having to dig through CSV files manually.

It includes:

- **Overview:** A quick look at sales activity and predicted risk.
- **Product Explorer:** Sales details for a selected product and store.
- **Risk Analysis:** Products ranked by their estimated likelihood of low future sales.
- **Model Performance:** Model comparisons and evaluation results.
- **Data Explorer:** A way to inspect and filter the available data.

The dashboard works with historical data, so these are not live sales predictions.

## Tools used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn
- Joblib

## Project structure

```text
Productpulse/
├── dashboard/       # Streamlit dashboard
├── data/            # Local dataset files
├── models/          # Saved model files
├── reports/         # Evaluation results
├── src/             # Data processing and training scripts
├── visualisation/   # Evaluation charts
├── .gitignore
├── requirements.txt
└── README.md
```

The raw and processed datasets are excluded from GitHub. The final trained model is also excluded because of its size.

## Running it on your computer

First, clone the repository:

```bash
git clone https://github.com/swaaley-07/Productpulse.git
cd Productpulse
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Download the M5 dataset from Kaggle and put the required CSV files in `data/raw/`.

Run the relevant scripts in `src/` to prepare the data and

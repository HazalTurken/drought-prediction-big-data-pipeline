# Drought Prediction Big Data Pipeline

## Overview
This project presents a scalable big data pipeline for drought prediction using **PySpark** and machine learning techniques. It processes large-scale environmental datasets to analyze patterns and estimate drought severity.

The pipeline demonstrates how distributed data processing can be used for real-world environmental analytics and decision support systems.

## Business Impact
This pipeline enables scalable environmental data analysis for early drought detection, supporting data-driven decision-making in agriculture, climate monitoring, and risk management.
---

## Key Features
- Distributed data processing with **PySpark**
- Data cleaning and preprocessing pipeline
- Feature engineering for drought analysis
- Aggregation and statistical analysis
- Scalable architecture for large datasets

---

## Dataset
Due to file size limitations, the full dataset is not included in this repository.

You can download the dataset from Kaggle:
https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data

Included sample files:
- `soil_data.csv`
- `geocodes_v2023.xlsx`

---

## Data Pipeline Steps
1. Load raw environmental datasets
2. Clean and preprocess missing values
3. Perform feature engineering
4. Merge datasets (time + soil + geolocation)
5. Compute drought-related metrics
6. Aggregate results by region/state
7. Export processed outputs

---
## Example Outputs
## Geospatial Drought Risk Analysis
<img width="2263" height="541" alt="image" src="https://github.com/user-attachments/assets/2b4c7239-0543-4b3f-a8a2-4d4f788bb741" />
This choropleth map visualizes the percentage of time each US county experiences severe drought conditions.

- Darker regions indicate higher drought risk
- Highlights spatial patterns across different regions
- Useful for identifying high-risk zones for climate impact

### Insights
- Western and southwestern regions show significantly higher drought exposure
- Central regions display moderate variability
- Eastern regions generally experience lower drought severity

## County-Level Drought Analysis (Nevada)
<img width="1239" height="664" alt="image" src="https://github.com/user-attachments/assets/eb256c04-9353-422c-9cb4-3298197163f9" />
This visualization shows the average drought severity score across counties in Nevada.  
The dashed line represents the state-wide average drought level, allowing comparison between regions.

## Time-Series Analysis (Streaming Results)
<img width="1403" height="491" alt="image" src="https://github.com/user-attachments/assets/f062cc2a-c0ef-437a-9c55-c1dc7e239df9" />
This visualization shows the relationship between precipitation levels and drought severity over time.

- Blue area: Average precipitation levels (mm)
- Red line: Average drought score (0–5)
- Timeline: Monthly observations

### Insights
- Lower precipitation levels generally correspond to higher drought severity
- Seasonal patterns can be observed in both precipitation and drought scores
- The model captures temporal trends useful for climate monitoring and forecasting

## Model Performance Evaluation (GBT)
<img width="581" height="558" alt="image" src="https://github.com/user-attachments/assets/1be0ff1b-ae15-462d-8858-cd1ed0ca57cf" />
This scatter plot compares actual drought scores with model predictions using Gradient Boosted Trees (GBT).

- Red dashed line represents perfect predictions (y = x)
- Points closer to the line indicate higher accuracy
- Spread indicates model error and variance

### Insights
- The model captures general trends but struggles with extreme values
- Predictions tend to cluster around average values
- Indicates potential need for additional feature engineering or temporal features


---

## Tech Stack
- Python
- PySpark
- Pandas
- NumPy
- Scikit-learn

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

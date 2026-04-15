# Installation of Java and PySpark
!apt-get install openjdk-11-jdk-headless -qq
!pip install pyspark==3.5.0

# Installation of OpenPyExcel
!pip install openpyxl

from pyspark.sql import SparkSession
import os, sys

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"


spark = (
    SparkSession.builder
    .appName("DroughtModeling")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.driver.extraJavaOptions", "-Djava.io.tmpdir=/tmp")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)

print("Spark version:", spark.version)

# Connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Read CSV
df_time = spark.read.csv("/content/drive/MyDrive/BigData/drought/train_timeseries.csv", header=True, inferSchema=True)
df_soil = spark.read.csv("/content/drive/MyDrive/BigData/drought/soil_data.csv", header=True, inferSchema=True)

df_time.show(5)
df_soil.show(5)

# Check record count and schema
print("df_time Record Count:", df_time.count())
print("df_time Schema:")
df_time.printSchema()

print("\ndf_soil Record Count:", df_soil.count())
print("df_soil Schema:")
df_soil.printSchema()

# Check duplicates from df_time
print(f"df_time original record count: {df_time.count()}")
df_time_deduplicated = df_time.dropDuplicates()
print(f"df_time deduplicated record count: {df_time_deduplicated.count()}")

# Remove duplicates from df_soil
print(f"\ndf_soil original record count: {df_soil.count()}")
df_soil_deduplicated = df_soil.dropDuplicates()
print(f"df_soil deduplicated record count: {df_soil_deduplicated.count()}")

from pyspark.sql.functions import col, sum

# Check null values in df_time
print("Null values in df_time:")
df_time.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_time.columns]).show()

# Check null values in df_soil
print("\nNull values in df_soil:")
df_soil.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_soil.columns]).show()

# Handling null score
print(f"df_time record count before dropping nulls: {df_time.count()}")
# Remove rows where the 'score' column is null
df_time_cleaned = df_time.dropna(subset=['score'])
print(f"df_time record count after dropping nulls: {df_time_cleaned.count()}")
print("""
Null values were removed because the missing values correspond to the drought score derived from NASA data,
which cannot be reliably estimated or imputed using other variables. Since this score is based on complex
satellite measurements and domain-specific calculations, attempting to fill missing values could introduce
significant bias and reduce the validity of the analysis. Therefore, records with null scores were excluded
to maintain data integrity and ensure accurate results.
""")

# Join Dataset
df_joined = df_time_cleaned.join(df_soil, on="fips", how="left")
print("Joined row count:", df_joined.count())
print("Joined columns:", df_joined.columns)

import matplotlib.pyplot as plt
import seaborn as sns

# Check outliers thru boxplot
columns_to_check = [
    'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN',
    'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M',
    'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE', 'score', 'elevation', 'slope1',
    'slope2', 'slope3', 'slope4', 'slope5', 'slope6', 'slope7', 'slope8',
    'aspectN', 'aspectE', 'aspectS', 'aspectW', 'aspectUnknown', 'WAT_LAND',
    'NVG_LAND', 'URB_LAND', 'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND',
    'CULT_LAND', 'SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7'
]

# Filter for columns that actually exist in df_joined and are numerical
existing_numerical_cols = [col_name for col_name in columns_to_check if col_name in df_joined.columns]

sample_ratio = 0.001 # 0.1% sample
df_sampled = df_joined.select(existing_numerical_cols).sample(False, sample_ratio, seed=42)

# Convert to Pandas DataFrame for plotting
pd_df_sampled = df_sampled.toPandas()

# Set up the plotting grid
n_cols = 4
n_rows = (len(existing_numerical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()

# Generate box plots for each column
for i, col_name in enumerate(existing_numerical_cols):
    if i < len(axes):
        sns.boxplot(y=pd_df_sampled[col_name], ax=axes[i])
        axes[i].set_title(col_name)
        axes[i].set_ylabel('') # Hide y-label to avoid clutter
    else:
        break # Stop if all axes are filled

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Outlier Detection using Box Plots (0.1% Sample of df_joined)', y=1.02, fontsize=16)
plt.show()

# Summary Statistics
print("Summary statistics of df_joined:")
df_joined.describe().show()

"""Although a significant proportion of outliers was identified (e.g., 23% in precipitation), these values were retained as they reflect natural environmental variability rather than anomalies. Environmental data are inherently skewed due to rare but impactful extreme events, and preserving these values is essential for accurate drought analysis."""

import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Load file for fips mapping
excel_file_path = "/content/drive/MyDrive/BigData/drought/geocodes_v2023.xlsx"

try:
    # Read the Excel file into a Pandas DataFrame
    pdf_fips_value = pd.read_excel(excel_file_path)
    print(f"Successfully loaded {excel_file_path} into Pandas DataFrame.")
    print("Pandas DataFrame head:")
    print(pdf_fips_value.head())

    # Infer schema from Pandas DataFrame to create Spark DataFrame
    df_fips_value = spark.createDataFrame(pdf_fips_value)

    print("\nSuccessfully converted to Spark DataFrame 'df_fips_value'.")
    print("Spark DataFrame schema:")
    df_fips_value.printSchema()
    print(f"Spark DataFrame count: {df_fips_value.count()}")

except FileNotFoundError:
    print(f"Error: The file '{excel_file_path}' was not found. Please check the path and filename.")
except Exception as e:
    print(f"An error occurred while loading the Excel file: {e}")

# Map fips code to State/Territory and County
df_featured_engineer = df_joined.join(df_fips_value, on="fips", how="left")
print("df_featured_engineer row count:", df_featured_engineer.count())
print("df_featured_engineer columns:", df_featured_engineer.columns)
df_featured_engineer.show(5)

print("Schema of df_featured_engineer:")
df_featured_engineer.printSchema()

# Check NULL Values
from pyspark.sql.functions import col, sum

print("Null values in df_featured_engineer:")
df_featured_engineer.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_featured_engineer.columns]).show()

# Identify fips with NULL values
from pyspark.sql.functions import col

print("Fips where other columns are NULL:")
df_featured_engineer.filter(col('State_Code').isNull()).select('fips').distinct().show()

# Imputation of NULL values based on available data
from pyspark.sql.functions import when, col, sum

# Update 'State_or_Territory', 'State_Code', 'State_Abbreviation', and 'County' where fips = 46102
df_featured_engineer = df_featured_engineer.withColumn(
    "State_or_Territory",
    when(col("fips") == 46102, "South Dakota").otherwise(col("State_or_Territory"))
).withColumn(
    "State_Code",
    when(col("fips") == 46102, 46).otherwise(col("State_Code"))
).withColumn(
    "State_Abbreviation",
    when(col("fips") == 46102, "SD").otherwise(col("State_Abbreviation"))
).withColumn(
    "County",
    when(col("fips") == 46102, 102).otherwise(col("County"))
)

# Verify the update for fips = 46102
# Recheck NULL values after imputation
print("Null values in df_featured_engineer:")
df_featured_engineer.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_featured_engineer.columns]).show()

df_featured_engineer.show(5)

# Average drought score per state or territory
from pyspark.sql.functions import avg

avg_score_per_state = df_featured_engineer.groupBy('State_or_Territory').agg(avg('score').alias('Average_Drought_Score'))


# Average land use per state or territory
from pyspark.sql.functions import avg, col

land_name_map = {
    "WAT_LAND": "Water",
    "NVG_LAND": "Natural Vegetation",
    "URB_LAND": "Urban",
    "GRS_LAND": "Grassland",
    "FOR_LAND": "Forest",
    "CULTRF_LAND": "Cultivated Rainfed",
    "CULTIR_LAND": "Cultivated Irrigated",
    "CULT_LAND": "Cultivated"
}

# Define the land_use_columns list
land_use_columns = [
    "WAT_LAND", "NVG_LAND", "URB_LAND", "GRS_LAND",
    "FOR_LAND", "CULTRF_LAND", "CULTIR_LAND", "CULT_LAND"
]

agg_exprs = [
    avg(col_name).alias(f"{land_name_map.get(col_name, col_name)}")
    for col_name in land_use_columns
]

# Group by State_or_Territory and calculate the average for each land use column
avg_land_use_per_state = df_featured_engineer.groupBy('State_or_Territory').agg(*agg_exprs)

avg_land_use_per_state.show(truncate=False)

"""Trend Visualisation"""

from pyspark.sql.functions import to_date, when, count, year, month

df_featured_engineer = df_featured_engineer.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

df_featured_engineer = df_featured_engineer.withColumn("year", year("date")).withColumn("month", month("date"))

#Check Feature Grouping & Correlation
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

meteo_cols = [
    'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'QV2M',
    'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE',
    'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE',
    'PS', 'PRECTOT'
]

assembler = VectorAssembler(inputCols=meteo_cols, outputCol="meteo_features")
df_vector = assembler.transform(df_featured_engineer).select("meteo_features")

matrix = Correlation.corr(df_vector, "meteo_features", "pearson").head()

corr_array = matrix[0].toArray()

corr_df = pd.DataFrame(corr_array, index=meteo_cols, columns=meteo_cols)

plt.figure(figsize=(16, 12))
sns.heatmap(corr_df,
            annot=True,
            cmap='coolwarm',  # Red = positive, Blue = negative correlation
            fmt=".2f",
            vmin=-1, vmax=1,
            linewidths=0.5)

plt.title("Meteorological Features Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Weather Profiling by Drought Severity
from pyspark.sql.functions import avg, round

df_featured_engineer = df_featured_engineer.withColumn("score_binned", round(df_featured_engineer["score"]))
severity_profile_spark = df_featured_engineer.groupBy("score_binned").agg(
    avg("PRECTOT").alias("avg_precipitation"),
    avg("T2M").alias("avg_temp"),
    avg("QV2M").alias("avg_humidity"),
    avg("WS10M").alias("avg_wind")
).orderBy("score_binned")

severity_profile_pd = severity_profile_spark.toPandas()

print(severity_profile_pd)

#Binned the score to show bar charts.
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Average Weather Conditions by Binned Drought Severity (0-5)', fontsize=16)

# Precipitation
sns.barplot(ax=axes[0, 0], data=severity_profile_pd, x='score_binned', y='avg_precipitation', hue='score_binned', palette='Blues_d', legend=False)
axes[0, 0].set_title('Average Precipitation (PRECTOT)')
axes[0, 0].set_ylabel('Precipitation (mm)')

# Temperature
sns.barplot(ax=axes[0, 1], data=severity_profile_pd, x='score_binned', y='avg_temp', hue='score_binned', palette='OrRd', legend=False)
axes[0, 1].set_title('Average Temperature (T2M)')
axes[0, 1].set_ylabel('Temperature (°C)')

# Humidity
sns.barplot(ax=axes[1, 0], data=severity_profile_pd, x='score_binned', y='avg_humidity', hue='score_binned', palette='Greens_d', legend=False)
axes[1, 0].set_title('Average Specific Humidity (QV2M)')
axes[1, 0].set_ylabel('Humidity')

# Wind
sns.barplot(ax=axes[1, 1], data=severity_profile_pd, x='score_binned', y='avg_wind', hue='score_binned', palette='Purples_d', legend=False)
axes[1, 1].set_title('Average Wind Speed (WS10M)')
axes[1, 1].set_ylabel('Wind Speed (m/s)')

plt.tight_layout()
plt.show()

#Time-Series
macro_trend_spark = df_featured_engineer.groupBy("year", "month").agg(
    avg("score").alias("avg_drought_score"),
    avg("PRECTOT").alias("avg_precipitation")
).orderBy("year", "month")

macro_trend_pd = macro_trend_spark.toPandas()

macro_trend_pd['date'] = pd.to_datetime(
    macro_trend_pd['year'].astype(str) + '-' + macro_trend_pd['month'].astype(str) + '-01'
)

fig, ax1 = plt.subplots(figsize=(16, 6))

# Drought Score (Red Line)
color = 'tab:red'
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Avg Drought Score', color=color, fontsize=12)
ax1.plot(macro_trend_pd['date'], macro_trend_pd['avg_drought_score'], color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

# Precipitation (Blue Area/Bar)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Avg Precipitation (mm)', color=color, fontsize=12)
ax2.fill_between(macro_trend_pd['date'], macro_trend_pd['avg_precipitation'], color=color, alpha=0.3)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Macro Trend: Drought Severity vs Precipitation Over Time', fontsize=16)
fig.tight_layout()
plt.show()

# Plot the time serious diagram with high drought score
from pyspark.sql.functions import col, avg, sum, when, count

# Group by FIPS and calculate severity metrics
county_severity_spark = df_featured_engineer.groupBy("State_or_Territory").agg(
    avg("score").alias("avg_score"),
    sum(when(col("score") >= 3, 1).otherwise(0)).alias("severe_weeks"),
    count("score").alias("total_weeks")
)

# Calculate the percentage of weeks spent in severe drought
county_severity_spark = county_severity_spark.withColumn(
    "pct_severe", (col("severe_weeks") / col("total_weeks")) * 100
)

# Bring the top 10 worst counties to Pandas to see our results
worst_counties_pd = county_severity_spark.orderBy(col("pct_severe").desc()).limit(10).toPandas()
print("Top 5 Drought-Prone Counties:")
print(worst_counties_pd)

worst_fips = worst_counties_pd['State_or_Territory'].iloc[0]
print(f"\nZooming in on State: {worst_fips}")

df_hotspot = df_featured_engineer.filter(col("State_or_Territory") == 'Nevada')

hotspot_macro_trend_spark = df_hotspot.groupBy("year", "month").agg(
    avg("score").alias("avg_drought_score"),
    avg("PRECTOT").alias("avg_precipitation")
).orderBy("year", "month")

hotspot_macro_trend_pd = hotspot_macro_trend_spark.toPandas()

hotspot_macro_trend_pd['date'] = pd.to_datetime(
    hotspot_macro_trend_pd['year'].astype(str) + '-' + hotspot_macro_trend_pd['month'].astype(str) + '-01'
)

hotspot_macro_trend_pd['rolling_drought'] = hotspot_macro_trend_pd['avg_drought_score'].rolling(window=6, min_periods=1).mean()
hotspot_macro_trend_pd['rolling_precip'] = hotspot_macro_trend_pd['avg_precipitation'].rolling(window=6, min_periods=1).mean()

fig, ax1 = plt.subplots(figsize=(16, 6))

color = 'tab:red'
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Avg Drought Score (6-Mo Rolling)', color=color, fontsize=12)
ax1.plot(hotspot_macro_trend_pd['date'], hotspot_macro_trend_pd['rolling_drought'], color=color, linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Avg Precipitation (6-Mo Rolling)', color=color, fontsize=12)
ax2.fill_between(hotspot_macro_trend_pd['date'], hotspot_macro_trend_pd['rolling_precip'], color=color, alpha=0.3)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Nevada Macro Trend:Drought Severity vs Precipitation (6-Month Rolling Average)', fontsize=16)
fig.tight_layout()

# Spatial/Geographical Aggregation
from pyspark.sql.functions import sum, when, col, count, lpad


# Cast the FIPS column to a string, then left-pad it with '0' until it is 5 characters long
df_padded = df_featured_engineer.withColumn("fips", lpad(col("fips").cast("string"), 5, "0"))


county_risk_spark = df_padded.groupBy("fips").agg(
    avg("score").alias("overall_avg_score"),
    sum(when(col("score") >= 3, 1).otherwise(0)).alias("weeks_in_severe_drought"),
    count("score").alias("total_weeks_recorded")
)

# Calculate the percentage of time spent in severe drought
county_risk_spark = county_risk_spark.withColumn(
    "pct_severe_drought",
    (col("weeks_in_severe_drought") / col("total_weeks_recorded")) * 100
)
county_risk_spark.show()

from urllib.request import urlopen
import json
import plotly.express as px
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

county_risk_pd = county_risk_spark.toPandas()

fig = px.choropleth(
    county_risk_pd,
    geojson=counties,
    locations='fips',
    color='pct_severe_drought',
    color_continuous_scale="YlOrRd",
    range_color=(0, county_risk_pd['pct_severe_drought'].max()),
    scope="usa",
    title="Historical Drought Risk by US County (% of time in Severe Drought)",
    labels={'pct_severe_drought': '% Time in Severe Drought'}
)

fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
fig.show()

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties_geo = json.load(response)

yearly_county_risk = df_padded.groupBy("year", "fips").agg(
    sum(when(col("score") >= 3, 1).otherwise(0)).alias("severe_weeks"),
    count("score").alias("total_weeks")
).withColumn(
    "pct_severe_drought",
    (col("severe_weeks") / col("total_weeks")) * 100
)

yearly_pd = yearly_county_risk.toPandas()

yearly_pd = yearly_pd.sort_values('year')

fig_animated = px.choropleth(
    yearly_pd,
    geojson=counties_geo,
    locations='fips',
    color='pct_severe_drought',
    color_continuous_scale="YlOrRd",
    range_color=(0, 60),
    scope="usa",
    animation_frame="year",
    title="Animated Historical Drought Risk by US County",
    labels={'pct_severe_drought': '% Severe'}
)

fig_animated.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
fig_animated.show()

"""# Drought Monitoring - Spark Structured Streaming
This notebook simulates real-time weather data ingestion using **Spark Structured Streaming**.  
The pipeline:
1. Splits train_timeseries.csv into small chunk files to simulate a live sensor feed  
2. Reads those files as a streaming source (one file = one micro-batch)  
3. Derives three analytical outputs:
   - **Console sink** - live drought-alert monitoring per micro-batch  
   - **Memory sink** - in-session aggregations queryable via Spark SQL and visualisations  
   - **Parquet sink** - durable storage of every processed batch

## I. Prepare Simulated Streaming Source
"""

df_clean = df_featured_engineer.select(
    "fips", "State_or_Territory", "County_Name", "date", "lat", "lon",
    "PRECTOT", "T2M", "T2M_MAX", "T2M_MIN", "WS10M", "score"
).dropna(subset=["score", "PRECTOT", "T2M_MAX"]).toPandas()

print("Clean row count:", df_clean.count())
df_clean.head(5)

import os, shutil, time
import pandas as pd

STREAM_IN  = "/content/stream_input"
STREAM_OUT = "/content/stream_output"
CHECKPOINT = "/content/checkpoints"

for p in [STREAM_IN, STREAM_OUT, CHECKPOINT]:
    shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p)

KEEP = ["fips", "State_or_Territory", "County_Name", "date", "PRECTOT", "PS", "QV2M", "T2M",
        "T2M_MAX", "T2M_MIN", "T2M_RANGE", "WS10M", "score"]
available = [c for c in KEEP if c in df_featured_engineer.columns]

df_main = df_clean
print(f"Rows from df_joined: {len(df_main):,}")

CHUNK_SIZE = 100000
n_chunks = 0
for i, start in enumerate(range(0, len(df_main), CHUNK_SIZE)):
    chunk = df_main.iloc[start:start + CHUNK_SIZE]
    chunk.to_csv(f"{STREAM_IN}/chunk_{i:04d}.csv", index=False)
    n_chunks += 1

print(f"Created {n_chunks} chunk files in {STREAM_IN}/")
print("Columns used:", available)

"""## II. Define the Streaming Schema
Providing an explicit schema is mandatory for Structured Streaming. It cannot infer schema from a live stream at runtime.
"""

stream_schema = StructType([
    StructField("fips",               StringType(), True),
    StructField("State_or_Territory", StringType(), True),
    StructField("County_Name",        StringType(), True),
    StructField("date",               StringType(), True),
    StructField("lat",                DoubleType(), True),
    StructField("lon",                DoubleType(), True),
    StructField("PRECTOT",            DoubleType(), True),
    StructField("T2M",                DoubleType(), True),
    StructField("T2M_MAX",            DoubleType(), True),
    StructField("T2M_MIN",            DoubleType(), True),
    StructField("WS10M",              DoubleType(), True),
    StructField("score",              DoubleType(), True),
])

"""## III. Create the Streaming DataFrame

"""

from pyspark.sql import functions as F

df_stream = (
    spark.readStream
         .format("csv")
         .option("header", "true")
         .option("maxFilesPerTrigger", 1)
         .schema(stream_schema)
         .load(STREAM_IN)
)

# 1. Add the filter
df_filtered = df_stream.filter(F.col("score").isNotNull())

# 2. Use df_filtered for your derived columns
df_enriched = df_filtered.withColumn(
    "drought_label",
    F.when(F.col("score") == 0, "D0 - No Drought")
     .when(F.col("score") == 1, "D1 - Abnormally Dry")
     .when(F.col("score") == 2, "D2 - Moderate Drought")
     .when(F.col("score") == 3, "D3 - Severe Drought")
     .when(F.col("score") == 4, "D4 - Extreme Drought")
     .when(F.col("score") == 5, "D5 - Exceptional Drought")
     .otherwise("Unknown")
).withColumn(
    "is_drought", F.when(F.col("score") >= 1, 1).otherwise(0)
).withColumn(
    "high_risk",  F.when(F.col("score") >= 3, 1).otherwise(0)
)

print("Streaming DataFrame ready. is Streaming:", df_enriched.isStreaming)

"""## Sink A: Console Output



"""

print("Streaming DataFrame is Streaming:", df_enriched.isStreaming)

"""## Sink B: Memory Sink"""

# Stop any existing streams to avoid conflicts
for s in spark.streams.active:
    s.stop()

# Memory Query 1 - regional drought summary
df_region = (
    df_enriched
    .groupBy("State_or_Territory", "County_Name")
    .agg(
        F.count("*").alias("total_obs"),
        F.round(F.avg("score"),    3).alias("avg_drought_score"),
        F.round(F.avg("PRECTOT"), 3).alias("avg_precip_mm"),
        F.round(F.avg("T2M_MAX"), 2).alias("avg_max_temp_c"),
        F.sum("is_drought").alias("drought_obs"),
        F.sum("high_risk").alias("high_risk_obs")
    )
)

query_region = (
    df_region.writeStream
    .outputMode("complete")
    .format("memory")
    .queryName("drought_by_region")
    .trigger(processingTime="3 seconds")
    .start()
)

# Memory Query 2 - monthly trend
df_trend = (
    df_enriched
    .withColumn("year_month", F.substring("date", 1, 7))
    .groupBy("year_month")
    .agg(
        F.count("*").alias("obs_count"),
        F.round(F.avg("score"),    3).alias("avg_drought_score"),
        F.round(F.avg("PRECTOT"), 3).alias("avg_precip_mm"),
        F.round(F.avg("T2M_MAX"),      2).alias("avg_temp_c"),
        F.sum("is_drought").alias("drought_obs")
    )
)

query_trend = (
    df_trend.writeStream
    .outputMode("complete")
    .format("memory")
    .queryName("drought_trend")
    .trigger(processingTime="3 seconds")
    .start()
)

# Memory Query 3 - high-risk alerts (append: only score >= 3)
query_alerts = (
    df_enriched
    .filter(F.col("high_risk") == 1)
    .select("State_or_Territory","County_Name","date","drought_label","score","PRECTOT","T2M_MAX","WS10M")
    .writeStream
    .outputMode("append")
    .format("memory")
    .queryName("high_risk_alerts")
    .trigger(processingTime="3 seconds")
    .start()
)

print("Memory sinks started: drought_by_region | drought_trend | high_risk_alerts")

import time

print("Waiting for memory streams to process all chunks.")

MAX_WAIT = 600
POLL     = 10

for elapsed in range(MAX_WAIT // POLL):
    time.sleep(POLL)
    progress = query_region.lastProgress
    batch_id = progress.get('batchId') if progress else 'not started'
    print(f"{elapsed * POLL}s elapsed. batchId: {batch_id}")

    if progress is not None and progress.get("batchId", -1) >= 20:
        statuses = [
            query_region.lastProgress,
            query_trend.lastProgress,
            query_alerts.lastProgress,
        ]
        if all(s is not None and s.get("numInputRows", 1) == 0 for s in statuses):
            print(f"All chunks processed after {elapsed * POLL}s - stopping early.")
            break

for q in [query_region, query_trend, query_alerts]:
    q.stop()
print("All memory streams stopped.")

"""### Regional Analysis"""

import matplotlib.pyplot as plt

region_df = spark.sql(
    'SELECT State_or_Territory, County_Name, avg_drought_score, avg_precip_mm, avg_max_temp_c,'
    ' drought_obs, high_risk_obs, total_obs,'
    ' ROUND(drought_obs   * 100.0 / total_obs, 1) AS drought_pct,'
    ' ROUND(high_risk_obs * 100.0 / total_obs, 1) AS high_risk_pct'
    ' FROM drought_by_region'
    ' WHERE total_obs >= 10'
    ' ORDER BY avg_drought_score DESC'
).toPandas()

print(f"Regions captured: {len(region_df)}")
print(region_df.head(10).to_string(index=False))

if region_df.empty:
    print("Skipping plot - region_df is empty. Fix the streaming step first.")
else:
    top20 = region_df.head(20)
    labels = top20['County_Name'] + ", " + top20['State_or_Territory']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(labels, top20['avg_drought_score'],
                 color='tomato', edgecolor='white')
    axes[0].set_xlabel('Average Drought Score (0-5)')
    axes[0].set_title('Top 20 Regions by Avg Drought Score')
    axes[0].invert_yaxis()
    overall_mean = region_df['avg_drought_score'].mean()
    axes[0].axvline(overall_mean, color='navy',
                    linestyle='--', label=f'Overall mean ({overall_mean:.2f})')
    axes[0].legend()

    axes[1].barh(labels, top20['high_risk_pct'],
                 color='darkorange', edgecolor='white')
    axes[1].set_xlabel('High-Risk Observations (%)')
    axes[1].set_title('Top 20 Regions by High-Risk Rate')
    axes[1].invert_yaxis()
    note = (
    "Note: Regions above the mean line with high high-risk % are priority monitoring zones.\n"
    )
    fig.text(0.01, -0.08, note, fontsize=8, color='dimgray', wrap=True)

    plt.tight_layout()
    plt.savefig('/content/streaming_regional_analysis.png', dpi=150)
    plt.show()


# Find the worst state and drill into its counties
worst_state = region_df.iloc[0]['State_or_Territory']


county_df = region_df[region_df['State_or_Territory'] == worst_state].copy()
county_df = county_df.sort_values('avg_drought_score', ascending=True)

fig, ax = plt.subplots(figsize=(12, max(5, len(county_df) * 0.4)))

bars = ax.barh(county_df['County_Name'], county_df['avg_drought_score'],
               color=plt.cm.YlOrRd(county_df['avg_drought_score'] / 5),
               edgecolor='white')

ax.axvline(county_df['avg_drought_score'].mean(), color='navy',
           linestyle='--', label=f"State mean ({county_df['avg_drought_score'].mean():.2f})")
ax.set_xlabel('Average Drought Score (0–5)')
ax.set_title(f'County-Level Drought Breakdown: {worst_state}')
ax.legend()
plt.tight_layout()
plt.savefig('/content/worst_state_county_breakdown.png', dpi=150)
plt.show()
print(f"Worst drought state: {worst_state}")
print(f"Insight: Counties above the dashed line are the most drought-stressed within {worst_state}.")

"""### Temporal Trend Analysis"""

# Get Trend over time using average
trend_df = spark.sql(
    'SELECT year_month, avg_drought_score, avg_precip_mm, avg_temp_c, obs_count'
    ' FROM drought_trend'
    ' WHERE year_month IS NOT NULL'
    ' ORDER BY year_month'
).toPandas()

print(f"Monthly periods captured: {len(trend_df)}")
print(trend_df.tail(10).to_string(index=False))

# Visualize temporal Trend analysis
fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()
x = range(len(trend_df))

ax1.fill_between(x, trend_df['avg_precip_mm'], alpha=0.35,
                 color='steelblue', label='Avg Precipitation (mm)')
ax1.set_ylabel('Avg Precipitation (mm)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2.plot(x, trend_df['avg_drought_score'], color='crimson',
         linewidth=2, label='Avg Drought Score')
ax2.set_ylabel('Avg Drought Score (0-5)', color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')
ax2.set_ylim(0, 5)

step = max(1, len(trend_df) // 12)
ax1.set_xticks(list(x)[::step])
ax1.set_xticklabels(trend_df['year_month'].iloc[::step], rotation=45, ha='right')
ax1.set_title('Monthly Drought Score vs Precipitation (Streaming Results)')

lines1, l1 = ax1.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, l1 + l2, loc='upper left')

plt.tight_layout()
plt.savefig('/content/streaming_temporal_trend.png', dpi=150)
plt.show()

# Seasonal heatmap from trend data
trend_df['year']  = trend_df['year_month'].str[:4]
trend_df['month'] = trend_df['year_month'].str[5:7]

heatmap_pivot = trend_df.pivot(index='year', columns='month', values='avg_drought_score')

month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
heatmap_pivot.columns = [month_labels[int(m)-1] for m in heatmap_pivot.columns]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(heatmap_pivot,
            cmap='YlOrRd',
            annot=True, fmt='.2f',
            linewidths=0.5,
            vmin=0, vmax=5,
            ax=ax)

ax.set_title('Average Drought Score by Year and Month (Streaming Results)')
ax.set_xlabel('Month')
ax.set_ylabel('Year')

note = (
    "Note: Dark cells reveal which year-month combinations experienced the worst drought conditions.\n"
    )
fig.text(0.01, -0.08, note, fontsize=8, color='dimgray', wrap=True)

plt.tight_layout()
plt.savefig('/content/streaming_seasonal_heatmap.png', dpi=150)
plt.show()

"""### High-Risk Alert Inspection
A closer look at D3–D5 observations captured by the alert stream.
"""

# Get alert stream from D3-D5 drought severity
alerts_df = spark.sql(
    'SELECT drought_label, COUNT(*) AS alert_count,'
    ' ROUND(AVG(score),    2) AS avg_score,'
    ' ROUND(AVG(PRECTOT), 3) AS avg_precip_mm,'
    ' ROUND(AVG(T2M_MAX), 2) AS avg_max_temp_c,'
    ' ROUND(AVG(WS10M),   2) AS avg_wind_ms'
    ' FROM high_risk_alerts'
    ' GROUP BY drought_label'
    ' ORDER BY avg_score DESC'
).toPandas()

print(alerts_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))  # wider figure
colors = ['#f4a261', '#e76f51', '#c1121f', '#800000']

# Filter out Unknown before plotting
alerts_clean = alerts_df[alerts_df['drought_label'] != 'Unknown']

bars = ax.bar(alerts_clean['drought_label'], alerts_clean['alert_count'],
              color=colors[:len(alerts_clean)], edgecolor='white')
ax.bar_label(bars, padding=3)
ax.set_ylabel('Alert Count')
ax.set_title('High-Risk Drought Alerts Captured During Streaming')

# fit the image
ax.set_ylim(0, alerts_clean['alert_count'].max() * 1.15)

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('/content/streaming_alert_distribution.png', dpi=150)
plt.show()

import numpy as np

alerts_time_pd = spark.sql(
    'SELECT drought_label, date'
    ' FROM high_risk_alerts'
    ' WHERE drought_label != "Unknown"'
).withColumn("year", F.year(F.to_date("date", "yyyy-MM-dd"))) \
 .groupBy("year", "drought_label") \
 .agg(F.count("*").alias("alert_count")) \
 .orderBy("year") \
 .toPandas()

pivot = alerts_time_pd.pivot(index="year", columns="drought_label", values="alert_count").fillna(0)

fig, ax = plt.subplots(figsize=(15, 6))
colors_map = {
    "D3 - Severe Drought":      "#2196F3",
    "D4 - Extreme Drought":     "#FF9800",
    "D5 - Exceptional Drought": "#B71C1C"
}
linestyles_map = {
    "D3 - Severe Drought":      "-",
    "D4 - Extreme Drought":     "--",
    "D5 - Exceptional Drought": "-."
}

for label in pivot.columns:
    ax.plot(pivot.index, pivot[label],
            label=label,
            color=colors_map.get(label, 'gray'),
            linestyle=linestyles_map.get(label, '-'),
            linewidth=2.5, marker='o', markersize=6)

ax.set_xticks(pivot.index)
ax.set_xticklabels(pivot.index, rotation=45, ha='right')
ax.set_xlabel('Year')
ax.set_ylabel('Total County-Week Records')
ax.set_title('High-Risk Drought Alert Frequency by Year (D3–D5)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

note = (
    "Note: Each data point represents the total number of county-week records classified under that drought level for the given year.\n"
    )
fig.text(0.01, -0.08, note, fontsize=8, color='dimgray', wrap=True)

plt.tight_layout()
plt.savefig('/content/streaming_alert_frequency.png', dpi=150, bbox_inches='tight')
plt.show()
print("Insight: Spikes in D4/D5 lines mark historically critical drought years requiring immediate response.")

"""## Sink C: Parquet Sink

"""

query_parquet = (
    df_enriched
    .writeStream
    .outputMode("append")
    .format("parquet")
    .option("path", STREAM_OUT)
    .option("checkpointLocation", CHECKPOINT)
    .trigger(processingTime="3 seconds")
    .queryName("parquet_sink")
    .start()
)

print("Parquet sink writing to:", STREAM_OUT)
query_parquet.awaitTermination(timeout=70)
query_parquet.stop()
print("Parquet sink finished")

# Copy parquet output to Google Drive
import shutil
shutil.copytree(
    "/content/stream_output",
    "/content/drive/MyDrive/stream_output",
    dirs_exist_ok=True
)
print("Parquet files copied to Google Drive")

"""#### Verify & Analyse Parquet Output"""

df_saved = spark.read.parquet(STREAM_OUT)

print(f"Total rows saved: {df_saved.count():,}")
df_saved.printSchema()

print('\nDrought label distribution in saved Parquet:')
df_saved.groupBy('drought_label').count().orderBy('drought_label').show(truncate=False)

# visualize parquet data
summary_pd = (
    df_saved
    .groupBy('drought_label')
    .agg(
        F.count('*').alias('count'),
        F.round(F.avg('PRECTOT'), 3).alias('avg_precip'),
        F.round(F.avg('T2M_MAX'), 2).alias('avg_max_temp'),
        F.round(F.avg('WS10M'),   2).alias('avg_wind')
    )
    .orderBy('drought_label')
    .toPandas()
)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # wider figure
for ax, col, label, color in zip(
    axes,
    ['avg_precip', 'avg_max_temp', 'avg_wind'],
    ['Avg Precipitation (mm)', 'Avg Max Temp (C)', 'Avg Wind Speed (m/s)'],
    ['steelblue', 'tomato', 'slategray']
):
    ax.bar(summary_pd['drought_label'], summary_pd[col],
           color=color, edgecolor='white', alpha=0.85)
    ax.set_ylabel(label)
    ax.set_title(label + ' by Drought Category')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Meteorological Conditions per Drought Category (from Parquet)', y=1.02)
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig('/content/streaming_parquet_summary.png', dpi=150, bbox_inches='tight')
plt.savefig('/content/drive/MyDrive/streaming_parquet_summary.png', dpi=150, bbox_inches='tight')
plt.show()

"""# **Machine Learning**"""

# Vector Assembler
from pyspark.ml.feature import VectorAssembler

feature_cols = [
    "lat", "lon",

    # Weather
    "PRECTOT", "T2M_MAX", "T2M_MIN", "WS10M",
    "QV2M", "PS", "T2MDEW", "T2M_RANGE",

    # Soil
    "elevation", "slope1",
    "aspectN", "aspectE",

    # Land
    "WAT_LAND", "NVG_LAND", "URB_LAND", "GRS_LAND",

    # Soil layers
    "SQ1", "SQ2", "SQ3"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

df_final = assembler.transform(df_featured_engineer).select("features", "score")
df_final.show(5, truncate=False)

# Train Test Split
train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

print("Train count:", train_data.count())
print("Test count:", test_data.count())

"""## **Random Forest**"""

# Random Forest Model
from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="score",
    predictionCol="prediction",
    numTrees=50,
    maxDepth=8,
    seed=42
)

rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)

rf_predictions.select("score", "prediction").show(10)

# Calculate RMSE for Random Forest
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol="score",
    predictionCol="prediction",
    metricName="rmse"
)

rmse_rf = evaluator.evaluate(rf_predictions)

#MAE
mae_eval = RegressionEvaluator(
    labelCol="score",
    predictionCol="prediction",
    metricName="mae"
)

mae_rf = mae_eval.evaluate(rf_predictions)

#R2
r2_eval = RegressionEvaluator(
    labelCol="score",
    predictionCol="prediction",
    metricName="r2"
)

r2_rf = r2_eval.evaluate(rf_predictions)

#MSE
mse_eval = RegressionEvaluator(
    labelCol="score",
    predictionCol="prediction",
    metricName="mse"
)

mse_rf = mse_eval.evaluate(rf_predictions)

print("Random Forest RMSE:", rmse_rf)
print("Random Forest MAE:", mae_rf)
print("Random Forest R²:", r2_rf)
print("Random Forest MSE:", mse_rf)

import pandas as pd

metrics_rf_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R²", "MSE"],
    "Value": [rmse_rf, mae_rf, r2_rf, mse_rf]
})

metrics_rf_df

# Actual vs Predicted
import matplotlib.pyplot as plt

# Using a small sample
pdf = rf_predictions.select("score", "prediction").sample(0.01).toPandas()

plt.figure(figsize=(6,6))
plt.scatter(pdf["score"], pdf["prediction"], alpha=0.3)

min_val = min(pdf["score"].min(), pdf["prediction"].min())
max_val = max(pdf["score"].max(), pdf["prediction"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted (Random Forest)")
plt.show()

# Feature importance
feature_cols = [
    "lat", "lon",
    "PRECTOT", "T2M_MAX", "T2M_MIN", "WS10M",
    "QV2M", "PS", "T2MDEW", "T2M_RANGE",
    "elevation", "slope1", "aspectN", "aspectE",
    "WAT_LAND", "NVG_LAND", "URB_LAND", "GRS_LAND",
    "SQ1", "SQ2", "SQ3"
]

importances = rf_model.featureImportances.toArray()
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=True)

plt.figure(figsize=(8, 8))
plt.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
plt.xlabel("Importance")
plt.title("Random Forest: Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

"""## **GBT Regressor**"""

# GBT Regressor
from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="score",
    predictionCol="prediction",
    maxIter=50,     # Tree count
    maxDepth=5,     # depth
    stepSize=0.1,   # learning rate
    seed=42
)

gbt_model = gbt.fit(train_data)
gbt_predictions = gbt_model.transform(test_data)

gbt_predictions.select("score", "prediction").show(10)

# Calculate RMSE for GBT Regressor
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol="score",
    predictionCol="prediction",
    metricName="rmse"
)

rmse_gbt = evaluator.evaluate(gbt_predictions)

#MAE
mae_eval = RegressionEvaluator(
    labelCol="score",
    predictionCol="prediction",
    metricName="mae"
)

mae_gbt = mae_eval.evaluate(gbt_predictions)

#R2
r2_eval = RegressionEvaluator(
    labelCol="score",
    predictionCol="prediction",
    metricName="r2"
)

r2_gbt = r2_eval.evaluate(gbt_predictions)

#MSE
mse_eval = RegressionEvaluator(
    labelCol="score",
    predictionCol="prediction",
    metricName="mse"
)

mse_gbt = mse_eval.evaluate(gbt_predictions)

print("RMSE:", rmse_gbt)
print("MAE:", mae_gbt)
print("R²:", r2_gbt)
print("MSE:", mse_gbt)

metrics_gbt_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R²", "MSE"],
    "Value": [rmse_gbt, mae_gbt, r2_gbt, mse_gbt]
})

metrics_gbt_df

# Actual vs Predicted
pdf_gbt = gbt_predictions.select("score", "prediction").sample(0.01).toPandas()

plt.figure(figsize=(6,6))
plt.scatter(pdf_gbt["score"], pdf_gbt["prediction"], alpha=0.3)

min_val = min(pdf_gbt["score"].min(), pdf_gbt["prediction"].min())
max_val = max(pdf_gbt["score"].max(), pdf_gbt["prediction"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted (GBT)")
plt.show()

# Feature importance
feature_cols = [
    "lat", "lon",
    "PRECTOT", "T2M_MAX", "T2M_MIN", "WS10M",
    "QV2M", "PS", "T2MDEW", "T2M_RANGE",
    "elevation", "slope1", "aspectN", "aspectE",
    "WAT_LAND", "NVG_LAND", "URB_LAND", "GRS_LAND",
    "SQ1", "SQ2", "SQ3"
]

importances_gbt = gbt_model.featureImportances.toArray()
importance_df_gbt = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances_gbt
}).sort_values("importance", ascending=True)

plt.figure(figsize=(8, 8))
plt.barh(importance_df_gbt["feature"], importance_df_gbt["importance"], color="steelblue")
plt.xlabel("Importance")
plt.title("GBT Regressor: Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
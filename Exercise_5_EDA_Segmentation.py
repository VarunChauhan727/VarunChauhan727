
# -------------------------------------------------------
# Exercise 5: Exploratory Data Analysis and Insight Generation
# Author: Varun Chauhan
# -------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set(style="whitegrid")

# -----------------------------
# Load Dataset
# -----------------------------
# Update the file path as needed
df = pd.read_csv("canada_post_campaign_data.csv")

# -----------------------------
# Target Variable
# -----------------------------
target_variable = "Response_Flag"
print(df[target_variable].value_counts(normalize=True))

# -----------------------------
# Correlation Analysis
# -----------------------------
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()[target_variable].sort_values(ascending=False)
print(correlation)

# Top correlated variables
top_corr = correlation.drop(target_variable).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_corr.values, y=top_corr.index)
plt.title("Top Variables Correlated with Response_Flag")
plt.show()

# -----------------------------
# Heatmap
# -----------------------------
top_vars = top_corr.index.tolist() + [target_variable]

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df[top_vars].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
for col in top_corr.index:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x=col, hue=target_variable, fill=True)
    plt.title(f"Distribution of {col} by Response")
    plt.show()

# -----------------------------
# Segmentation (K-Means)
# -----------------------------
cluster_vars = top_corr.index.tolist()
df_cluster = df[cluster_vars].fillna(df[cluster_vars].mean())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cluster)

inertia = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 8), inertia, marker="o")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df["Segment"] = kmeans.fit_predict(scaled_data)

print(df["Segment"].value_counts())

segment_profile = df.groupby("Segment")[cluster_vars + [target_variable]].mean()
print(segment_profile)

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:57:02 2024

@author: jayan
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
import statsmodels.api as sm

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to extract and clean data for a specific indicator and year
def slice(df, indicator, year):
    """
    Extract and clean data for a specific indicator and year from the given DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the data.
    - indicator (str): Indicator name to extract.
    - year (str): Year to extract.

    Returns:
    - pd.DataFrame: Cleaned DataFrame with selected indicator and year for each country.
    """
    df_slice = df[(df['Indicator Name'] == indicator)].reset_index(drop=True)
    return df_slice[['Country Name', year]].rename(columns={year: indicator})

# Function to merge and clean data for two indicators and a specific year
def merge_and_clean(indicator1, indicator2, year, df):
    """
    Merge and clean data for two indicators and a specific year.

    Parameters:
    - indicator1 (str): First indicator name.
    - indicator2 (str): Second indicator name.
    - year (str): Year to extract.
    - df (pd.DataFrame): Input DataFrame containing the data.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: Cleaned DataFrame and its transposed version.
    """
    data1 = slice(df, indicator1, year)
    data2 = slice(df, indicator2, year)
    merged_df = pd.merge(data1, data2, on='Country Name', how='outer')
    clean_df = merged_df.dropna(how='any').reset_index(drop=True)
    tran_df = clean_df.transpose()
    tran_df.columns = tran_df.iloc[0]
    tran_df = tran_df.iloc[1:]
    return clean_df, tran_df

# Function to calculate inertia for KMeans clustering and generate elbow plot
def calculate_inertia(data, max_clusters=10):
    """
    Calculate inertia for KMeans clustering and generate elbow plot.

    Parameters:
    - data (pd.DataFrame): Input DataFrame for clustering.
    - max_clusters (int): Maximum number of clusters for the elbow plot.

    Returns:
    - list: List of inertia values for each number of clusters.
    """
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

# Function to perform KMeans clustering and plot the results for two different years
def perform_clustering(X, Y):
    """
    Perform KMeans clustering and plot the results for two different years.

    Parameters:
    - X (pd.DataFrame): DataFrame for the first year.
    - Y (pd.DataFrame): DataFrame for the second year.

    Returns:
    - None
    """
    kmeans_X = KMeans(n_clusters=4, random_state=42)
    X['cluster'] = kmeans_X.fit_predict(
        X[['Population growth (annual %)', 'CO2 emissions (kt)']])
    X['cluster'] = X['cluster'].astype('category')

    kmeans_Y = KMeans(n_clusters=4, random_state=42)
    Y['cluster'] = kmeans_Y.fit_predict(
        Y[['Population growth (annual %)', 'CO2 emissions (kt)']])
    Y['cluster'] = Y['cluster'].astype('category')

    cluster_centers_X = kmeans_X.cluster_centers_
    cluster_centers_Y = kmeans_Y.cluster_centers_

    cluster_palette = sns.color_palette("husl", 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scatter_plot_X = sns.scatterplot(x='Population growth (annual %)',
                                     y='CO2 emissions (kt)', hue='cluster', 
                                     data=X, palette=cluster_palette, ax=ax1)
    ax1.scatter(cluster_centers_X[:, 0], cluster_centers_X[:, 1],
                marker='h', s=50, c='black', label='Cluster Centers')
    ax1.set_title('Population Growth and CO2 Emission in 2000')
    ax1.legend(title='Clusters', loc='upper right')

    scatter_plot_Y = sns.scatterplot(x='Population growth (annual %)',
                                     y='CO2 emissions (kt)', hue='cluster',
                                     data=Y, palette=cluster_palette, ax=ax2)
    ax2.scatter(cluster_centers_Y[:, 0], cluster_centers_Y[:, 1],
                marker='h', s=50, c='black', label='Cluster Centers')
    ax2.set_title('Population Growth and CO2 Emission in 2020')
    ax2.legend(title='Clusters', loc='upper right')

    plt.tight_layout()
    plt.show()

# Function to forecast and plot data for selected countries with confidence intervals
def forecast_and_plot_with_confidence(df, selected_countries, indicator_name):
    """
    Forecast and plot data for selected countries with confidence intervals.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the data.
    - selected_countries (list): List of countries to forecast and plot.
    - indicator_name (str): Name of the indicator to forecast.

    Returns:
    - None
    """
    data_selected = df[(df['Country Name'].isin(selected_countries)) & (
        df['Indicator Name'] == indicator_name)].reset_index(drop=True)
    data_forecast = data_selected.melt(
        id_vars=['Country Name', 'Indicator Name'], var_name='Year',
        value_name='Value')
    data_forecast = data_forecast[data_forecast['Year'].str.isnumeric()]
    data_forecast['Year'] = data_forecast['Year'].astype(int)
    data_forecast['Value'].fillna(data_forecast['Value'].mean(), inplace=True)
    data_forecast = data_forecast[(data_forecast['Year'] >= 1990) & (
        data_forecast['Year'] <= 2020)]

    predictions = {}
    all_years_extended = list(range(1990, 2026))

    for country in selected_countries:
        plt.figure(figsize=(6, 4))

        # Actual Data
        plt.plot(data_forecast[data_forecast
                               ['Country Name'] == country]['Year'],
                 data_forecast[data_forecast['Country Name']
                               == country]['Value'],
                 marker='.', label=f'Actual Data', color='green')

        country_data = data_forecast[data_forecast['Country Name'] == country]
        X_country = country_data[['Year']]
        y_country = country_data['Value']
        degree = 3
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X_country)

        model = sm.OLS(y_country, X_poly).fit()

        # Prediction
        X_pred = poly_features.transform(
            pd.DataFrame(all_years_extended, columns=['Year']))
        forecast_values = model.predict(X_pred)

        # Confidence Interval
        conf_int = model.get_prediction(X_pred).conf_int(alpha=0.05)

        plt.plot(all_years_extended, forecast_values,
                 label='Best Fitting Curve', linestyle='-', color='skyblue')

        # Plot Confidence Interval
        plt.fill_between(all_years_extended, conf_int[:, 0], conf_int[:, 1],
                         color='orange', alpha=0.3, 
                         label='95% Confidence Interval')

        prediction_2025 = forecast_values[-1]
        plt.plot(2025, prediction_2025, marker='o', markersize=8,
                 label=f'Prediction for 2025 - {prediction_2025:.2f}', 
                 color='red')

        plt.title(
            f'{indicator_name} Forecast for {country} with Confidence Interval')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.xlim(1990, 2030)
        plt.xticks(range(1990, 2031, 5))
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
df = pd.read_csv('API_19_DS2_en_csv_v2_5998250.csv', skiprows=3)
selected_countries = ['India', 'United Kingdom', 'Japan']
indicator_name = 'CO2 emissions (kt)'

# Clustering
indicator1 = 'Population growth (annual %)'
indicator2 = 'CO2 emissions (kt)'
year1 = '2000'
year2 = '2020'
result_2000, result_2000_tran = merge_and_clean(
    indicator1, indicator2, year1, df)
result_2020, result_2020_tran = merge_and_clean(
    indicator1, indicator2, year2, df)
X = result_2000[['Population growth (annual %)', 'CO2 emissions (kt)']]
Y = result_2020[['Population growth (annual %)', 'CO2 emissions (kt)']]

# Calculate and plot inertia for elbow plot
inertias_X = calculate_inertia(
    X[['Population growth (annual %)', 'CO2 emissions (kt)']])
inertias_Y = calculate_inertia(
    Y[['Population growth (annual %)', 'CO2 emissions (kt)']])

plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), inertias_X, marker='o', label='2000', color='blue')
plt.plot(range(1, 11), inertias_Y, marker='o', label='2020', color='green')
plt.title('Elbow Plot for KMeans Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.legend()
plt.grid(True)
plt.show()

# Perform clustering and plot results
perform_clustering(X, Y)

# Forecasting and Plotting with Confidence Intervals
forecast_and_plot_with_confidence(df, selected_countries, indicator_name)

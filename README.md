# House Price Modelling: A Journey into Machine Learning and Web Scraping

This repository contains the code and analysis for a house price modelling project that leverages machine learning and advanced web scraping techniques. The goal of this project is to predict house prices using real-world data collected from Rightmove, with a particular focus on properties in Suffolk.

---

## Table of Contents

- [Overview](#overview)
- [Data Collection](#data-collection)
- [Data Exploration & Visualisation](#data-exploration--visualisation)
- [Data Preparation & Feature Engineering](#data-preparation--feature-engineering)
- [Model Development & Evaluation](#model-development--evaluation)
- [Advanced Models](#advanced-models)
- [Future Applications](#future-applications)


---

## Overview

As a budding data scientist, I challenged myself to build a predictive model for house prices using real-time, real-world data. Instead of relying solely on complete Kaggle datasets, I opted to scrape data from Rightmove, thereby enhancing my web scraping skills and creating a model with broader applicability beyond London's unique housing market.

**Key Achievements:**

- Developed an advanced web scraper to extract extensive numeric data from Rightmove listings.
- Implemented feature engineering techniques (including handling new homes and categorical data encoding) to improve model performance.
- Achieved an impressive R-squared score of **0.897** on Suffolk housing data using AutoGluon and XGBoost models.

---

## Data Collection

- **Source:** Rightmove (active and sold property listings)
- **Method:** 
  - Started with a basic GitHub repository for Rightmove scraping.
  - Analysed Rightmove's HTML structure to locate JSON data embedded within JavaScript.
  - Utilised the `lxml` library (specifically the `etree` function) to parse HTML and extract additional numeric data.
- **Data Points Collected:** 
  - Initial points: price, type, address, postcode, and number of bedrooms.
  - Expanded to include ten additional numeric data points per house, such as the number of images, floorplans, and virtual tours.
- **Special Handling:** 
  - Developed separate functions to handle sold properties alongside active listings.
  - Identified new homes by comparing Rightmove's 'new homes' search results with the overall dataset.

---

## Data Exploration & Visualisation

- **Initial Analysis:** 
  - Conducted exploratory data analysis (EDA) to identify anomalies and patterns.
- **Visualisations Created:**
  - **Average Price Heatmap:** Displayed property price distribution across the local area.
  - **Bedroom-to-Average-Price Bar Graph:** Highlighted trends and potential data issues.
  
These visual tools were essential in guiding subsequent data cleaning and feature engineering steps.

---

## Data Preparation & Feature Engineering

- **Dataset Focus:** Suffolk house listings (approximately 7,000 properties).
- **Selected Features:**
  - Numeric: bedrooms, bathrooms, numberOfImages, numberOfFloorplans, numberOfVirtualTours, latitude, longitude.
  - Categorical: property type (encoded using `sklearn`'s `LabelEncoder`).
  - Binary: new home status (1 for new homes, 0 otherwise).
- **Feature Engineering:**
  - Introduced new features by multiplying bedrooms and bathrooms with new home status to capture the premium associated with new builds.
  - Experimented with various data scaling techniques (which did not significantly improve model performance).

---

## Model Development & Evaluation

- **Baseline & Advanced Models:**
  - Initially experimented with a Random Forest model.
  - Explored advanced methods like Google's TabNet.
- **AutoML Toolkit:**
  - Leveraged **AutoGluon** to automate model training and comparison.
  - **XGBoost** emerged as the best-performing model, achieving an R-squared score of **0.897** on the Suffolk housing data.
- **Real-World Validation:**
  - Scraped 15 new house listings from Rightmove post data collection.
  - Used the trained model to predict their prices, offering valuable insights into model strengths and weaknesses.

---

## Advanced Models

While Random Forest and TabNet were considered during the experimentation phase, the integration of AutoGluon allowed for an efficient model comparison process. XGBoost was found to consistently outperform the alternatives, validating the approach for this particular dataset and problem.

---

## Future Applications

- **Model Refinement:** Further tuning and inclusion of additional features to improve prediction accuracy.
- **Scalability:** Adapting the web scraper and model to handle larger datasets and more geographic regions.
- **Deployment:** Building an interactive web application to provide real-time house price predictions for end-users.

---

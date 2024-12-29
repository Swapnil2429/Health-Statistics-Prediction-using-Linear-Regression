# **Health Statistics Prediction using Linear Regression**

## **Project Overview**
The goal of this project is to be able to estimate several types of health statistic using the factors such as prevalence, accessibility to the health facilities, populations affected, and other facts related to corresponding countries. Here, I conduct machine learning using a dataset of health statistics all over the world and **Linear Regression** to estimate a variable linked to health data.

## **Objective**
**Forecast** the occurrence probability of said rates or other health parameters depending on the chosen characteristics, including the country, year, or treatment kinds and others.

**Evaluate model performance** based on accuracy measures such as Mean Squared Error (MSE) and total accuracy rating (R² score).
There are several

**data pre- processing steps** that need to be followed in order to get the data into a format that is appropriate for modeling.

## **Dataset Description**
The dataset includes the following columns:
- **Country**: The country in which the health status data was collected from.
- **Year**: The year of data collection.
- **Disease Name**: It is the commonly used name of the disease or health condition.
- **Prevalence Rate (%)**: The proportion of the population contracting the disease.
- **Incidence Rate (%)**: New or newly diagnosed cases as a proportion of total incidence.
- **Mortality Rate (%)**: Ratio of the total number of people who are affected by the disease with the number of people dies from the same disease.
- **Age Group**: The age group that is most impacted by the disease.
- **Gender**: The affected gender(s) (Male, Female, Both).
- **Healthcare Access (%)**: The proportion of the population which has an access to healthcare.
- **Doctors per 1000**: Doctors to population density where the number of doctors per thousand people is presented.
- **Hospital Beds per 1000**: Availability of hospital beds per 1000 population.
- **Treatment Type**: Along with the primary treatment type, the specific form of treatment such as medication or surgery.
- **Average Treatment Cost (USD)**: The overall cost per patient of the disease.
- **Recovery Rate (%)**: A proportion of the population that gets cured of the disease.
- **DALYs**: Disability Adjusted Life Year Disablement Adjusted Life Year.
- **Improvement in 5 Years (%)**: A remedy to the disease has been seen to have been developed over the past five years.
- **Per Capita Income (USD)**: Income per capita of the country.
- **Education Index**: The average level of education within the country.
- **Urbanization Rate (%)**: Proportion of the population residing in the urban areas.

## **Installation and Setup**
To run this project, ensure that you have the following dependencies installed:
1. Install **Python** (version 3.7 or higher).

## **Data Preprocessing**
**The following steps were performed to clean and preprocess the data**:

 **Handling Missing Data**
Where there were missing attributes or variables, these were imputed if they were necessary or else the records having missing conditional attributes were deleted.

**Feature Encoding**
We have used One-Hot Encoding to encode categorical features, which include Country, Disease Name, and Age Group.

**Normalization**:

The basic numeric characteristics like Prevalence Rate (%), Incidence Rate (%), and Healthcare Access (%) were normalized in StandardScaler form.

**Data Splitting**:

This dataset was further divided into the training and test data set for the assessment of model performance.

## **Model Training**
We used Linear Regression to model the relationship between the features and the target variable. The following steps were followed:

**Splitting Data**:
The data was divided into features (X) and target (y), with training and testing datasets.

**Linear Regression Model**:
A linear regression model was trained on the training data using the fit() method.

**Model Evaluation**:
The model's performance was evaluated using Mean Squared Error (MSE) and R² score.
## **Results**

**Mean Squared Error (MSE)**: **3.6964e-23**

**R² Score**: **1.0**   
## **Conclusion**
As can be seen from this project the application of Linear Regression in predicting health statistics is effective. The model has produced very good results on the given dataset and it is seen that there is need to go for further subsequent steps to avoid overfitting of a model.

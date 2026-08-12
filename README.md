# IDXExchange Data Science Internship Summer 2026
Lydia Chin DS55

California Property Close Price Prediction Model


## Project Overview

This project aims to develop and assess machine learning models capability of predicting a property`s market value based on its characteristics. The models use historical California Regional Multiple Listing Service (CRMLS) data containing property information including its characteristics, amenities, location and associated features like school districts, and financial history. The project follows an end-to-end data science pipeline: exploring the data, preparing and cleaning it, engineering relevant features, building models, tuning hyperparameters, and evaluating results.

The following models are tested and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost

Model quality is measured using R², MAPE, and MdAPE. 

The quality of each model is measured using R², MAPE, and MdAPE scores. The final output of this project consists of a trained predictive model, supporting documentation, and a presentation covering the key insights and model results.

## Dataset Source
The project used 13 months of CRMLS property data. The most recent month, June 2026, was use as the testing set while data from May 2026 to May 2025 was used for training and validation. 
The target feature was `ClosePrice`, the final amount paid to the seller for a property. The model also only uses properties that satisfy the following conditions:

- `StateOrProvince = CA`
- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`
- `ClosePrice > 0`


## Preprocessing
1. All the columns containing leakage columns and other irrelevant information like real estate agent names and offices. These columns were:

    `BuyerAgentAOR`, `ListAgentAOR`, `OriginalListPrice`, `ListAgentFirstName`, `ListAgentLastName`,`ListAgentEmail`, `CoListAgentFirstName`, `CoListOfficeName`, `CoListAgentLastName`,
    `BuyerAgentMlsId`, `BuyerOfficeName`, `BuyerAgentFirstName`, `BuyerAgentLastName`, `BuyerOfficeAOR`, `CoBuyerAgentFirstName`, `ListAgentFullName`, `ListOfficeName`, 
    `LotSizeDimensions`, `LotSizeAcres`, `LotSizeArea`, `BuildingAreaTotal`, `ListPrice`, `OriginalListPrice`, `DaysOnMarket`

2. Columns with 95% or more missing values were dropped. These columns were:

    `MiddleOrJuniorSchoolDistrict`, `TaxAnnualAmount`, `CoveredSpaces`,`BusinessType`, `ElementarySchoolDistrict`, `FireplacesTotal`,
    `TaxYear`, `AboveGradeFinishedArea`, `WaterfrontYN`, `BelowGradeFinishedArea`, `BasementYN`, `BuilderName`

3. Rows with values that aren't possible like 0 bedrooms or bathrooms were dropped and any properties with a closing price greater than the 99th percentile were removed to decrease outliers

4. To deal with the rest of the missing values, missing values in categorical columns were marked as "Unknown" along with missingness flags while in numerical columns, median imputation was used. Boolean columns had their missing values filled in with a 0 as it is reasonable to conclude that a missing value for an amenity means that property doesn't have one. 

5. Categorical columns were encoded using One Hot Encoding in order to use them for regression models. Since some columns had a lot of cardinality, only the 200 most common entries were encoded with the rest identified as "Other"

6. Features `LotLivingRatio`, `BathroomBedroomRatio`, `TotalAmenityCount`, `FlooringCount` were added from existing columns and school district data was added using the California School District boundary GeoJSON from https://data.ca.gov/dataset/california-school-district-areas-2025-26


## Modeling
First, a baseline Linear Regression model was used followed by Decision Tree and Random Forest models. Then, an XGBoost model was explored and tuned using a random split of training data for training and validation with June 2026 data saved for testing the final selected model.

| Model             |   Training R2 |   Testing R2 |      MAE |    RMSE |    MAPE |    MdAPE |
|:------------------|-----------:|----------:|---------:|--------:|--------:|---------:|
| Linear Regression |   0.828333 |  0.785506 | 169925   | 521.471 | 21.738  | 13.3699  |
| Decision Tree     |   0.659456 |  0.670616 | 227005   | 580.5   | 28.0595 | 19.8884  |
| Random Forest     |   0.987195 |  0.908555 | 100074   | 421.373 | 10.4828 |  6.94778 |
| **XGBoost**       |   0.968902 | **0.914349** |  98132.8 | 414.534 | 10.28   |  7.02665 |

The final selected model is the XGBoost model as it returned the highest r-squared score along with the lowest MAE, RMSE, and MAPE scores.

*R-squared computes explained variation, MAE/RMSE compute dollar error in predictions(with RMSE emphasizing large differences), and MAPE/MdAPE computes relative error



## Reproduce Final Models
To reproduce the project, first download all CRMLS data from May 2025 to June 2026 and California District Geodata and add filepaths to final_model.py
Then follow the below steps to duplicate the same environment and run the script final_model.py which contains all work done.

```bash
conda env create -f environment.yml
conda activate idx
python final_model.py
```


## Repository Key

| Location | Purpose |
| --- | --- |
| `week2/01_exploration.ipynb` | Access source, distribution exploration |
| `week3/02_preprocessing.ipynb` | Filtering and cleaning |
| `week4/03_baseline_model.ipynb` | Linear Regression baseline model|
| `week5/04_model_comparison.ipynb` | Decision Tree and Random Forest comparison with baseline Regression |
| `week6/05_feature_engineering.ipynb` | Engineered features and school-district spatial join |
| `week7/05_advanced_models.ipynb` | Final reproducible XGBoost workflow with June 2026 evaluation |

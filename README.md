# Sales Prediction using Machine Learning Models

## Objective
The objective of this project is to predict the sales of each product by understanding product properties and outlet sales using machine learning models.

## Setup
- Python libraries: pandas, numpy, matplotlib, seaborn, scipy, sklearn, statsmodels, xgboost
- Dataset: Train big mart.csv
## Code Overview
1. Data Cleaning:
   - Handling missing values in 'Item_Weight' and 'Outlet_Size'.
   - Imputing missing 'Item_Weight' values based on mean values of different product types.
   - Imputing missing 'Outlet_Size' values based on mode values of different outlet types.
   - Replacing 0 values in 'Item_Visibility' with mean value.
   - Standardizing 'Item_Fat_Content' values (e.g., 'LF' to 'Low Fat', 'low fat' to 'Low Fat', 'reg' to 'Regular').

2. Feature Engineering:
   - Creating a new column 'New_Item_Type' based on the first two words of 'Item_Identifier'.
   - Modifying 'Item_Fat_Content' for Non-Consumable items.
   - Creating a new column 'Shelf_Life' to categorize items into Perishable and Non-Perishable.
   - Creating 'MRP_per_unit_weight' by dividing 'Item_MRP' by 'Item_Weight'.
   - Creating 'Outlet_years' by subtracting 'Outlet_Establishment_Year' from 2013.

3. Exploratory Data Analysis:
   - Visualizing numerical features using histograms and distribution plots.
   - Visualizing categorical features using count plots.
   - Analyzing sales trends based on different factors (e.g., Outlet Location Type, Outlet Type, Outlet Size, Item Type).

4. Modeling:
   - Label Encoding and One-Hot Encoding categorical variables.
   - Splitting the data into training and testing sets.
   - Building and evaluating different models:
     - Linear Regression
     - Random Forest Regression
     - XG Boost Regression

5. Model Evaluation:
   - Calculating RMSE, MAE, and cross-validation scores for each model.
   - Comparing actual sales vs. predicted sales for the best-performing model (XG Boost).

## Conclusion
- The XG Boost regressor performed the best with an R-squared score of 72%.
- Further hyperparameter tuning and model optimization can be explored to improve performance.

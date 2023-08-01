To download the dataset https://1drv.ms/u/s!AvPxibfbWBItkVlc3dXoQb1k2MaD?e=1zXQRQ

# Online travel agency: Data analysis / Machine Learning project

This project is using data from an online travel agency (OTA) to predict whether a user will book a hotel after viewing and clicking on the listing. The data has various features, such as user location, hotel rating, price, etc. I used Python to clean, analyze, and model the data.

## Data cleaning and Analysis

I performed some basic statistics and correlation analysis on the data. I found that some features were highly skewed, had many outliers, or had constant values. I also found that some features had a lot of missing values. I dropped 53% of the features that had more than 30% missing values or were redundant. I imputed the missing values for the remaining features using statistical measures.

## Click-through rate and Conversion rate

I calculated the click-through rate (CTR) and conversion rate (CVR) for the data. The CTR is 4.46%, meaning that only 4.46% of the users who viewed the hotel listing clicked on it. The CVR is 62.57%, meaning that 62.57% of the users who clicked on the hotel listing booked it.

# Model Training

I split the data into 80% train and 20% test sets and normalized the features. I used Logistic regression and CatBoost to predict the "booked" variable. I tuned the CatBoost hyperparameters with grid search. The best ones are: {'depth': 6, 'l2_leaf_reg': 1, 'learning_rate': 0.1}.

The CatBoost model has high precision, recall, f1-score and accuracy for both classes. It can correctly predict 99.39% of the test instances. The model works well on this data set.

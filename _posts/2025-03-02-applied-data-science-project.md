---
layout: post
author: Ching Chriselle
title: "Applied Data Science Project"
categories: ITD214
---
## Project Background
### Industry Landscape:
The Beauty and Personal Care industry is projected to exceed US$100bil in revenues in 2025. In the next 5 years, annual industry growth is estimated to hit 2.9% [[source](https://www.statista.com/outlook/cmo/beauty-personal-care/united-states)]. More importantly, sales from online avenues are expected to account for about 65.9% of 2025 revenues. SEPHORA is a leading beauty and personal care products retailer. It offers a catalogue of brand and housebrand goods across brick-and-mortar stores as wellas online. In 2022, its revenues from the United States reached almost US$7bil, and accounted for more than half of its global revenues [[source](https://www.statista.com/statistics/1445562/retail-sales-of-sephora-globally/)].

### Business Goal:
To improve online channel sales at Sephora

### Objectives:
1. To identify key factors driving product prices
2. To predict customer sentiment towards beauty products based on review text

## Work Accomplished
This project explores how objective 1 can be achieved to meet Sephora's business goal. 

### Data Source
Information on products and customer reviews on its skincare catalogue were scrapped from SEPHORA’s online store (collected as of Mar 2023) and published in KAGGLE [[source](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/)]. Dataset:
- Consists of 6 csv data files for over 8,000 products and more than 1 million customer reviews.
- Includes 1 file “product_info” that provides product information and 5 files “reviews...” that provide customer reviews on skincare products. Both groups of files can be joined by a unique assigned product identifier “product_id”.
- Includes information like product details, prices, ratings, ingredients, user free-text reviews, etc.

### Data Preparation
Objective 1 uses only data from "product_info". The data file contains 27 columns and over 8,000 rows of numerical and categorical values. Each row points to each distinct product that can be identified by its "product_id". 

**Dataframe:**

![image](https://github.com/user-attachments/assets/bd11a90b-4ed5-428b-aaf8-7a6c5633fd25)

After data exploration, data cleaning was applied to 3 key problem areas outlined below.

#### 1. Sparse Columns with Null Values
Several variables had many null values. They were either imputed by logic, or new features were engineered to replace such columns.

![image](https://github.com/user-attachments/assets/4399cab3-823c-4b93-8d2a-d5b332c08d11)

**Imputation by Logic**
- Imputed null values in "tertiary_category" as “Unknown” to demonstrate that these products do not have such in-depth categorical differentiation.
```python
df['tertiary_category'] = df['tertiary_category'].fillna('Unknown')
```
- There are 8 empty values in "secondary_category". They were imputed variedly below.
```python
if not missing_secondary_category_df.empty:
    # Get the most common secondary_category for Fragrance and impute corresponding index
    most_common_fragrance_category = df[df['primary_category'] == "Fragrance"]['secondary_category'].mode()[0]
    df.loc[df.index == 986, 'secondary_category'] = most_common_fragrance_category
    # For the following products, imputation of secondary_category by logic after referencing product name
    df.loc[df.index == 2043, 'secondary_category'] = "Fragrance"
    df.loc[df.index == 6661, 'secondary_category'] = "Cleansers"
    df.loc[df.index.isin([6781, 6782, 6784, 6785]), 'secondary_category'] = "Gift Card"
    df.loc[df.index == 8190, 'secondary_category'] = "Mini Size"
```
- Zeroised null values in "ratings" and "reviews" to indicate that products were not rated or reviewed.
```python
df['rating'] = df['rating'].fillna(0)
df['reviews'] = df['reviews'].fillna(0)
```

**Feature Engineering**
Some price-related variables e.g. "sale_price_usd", "value_price_usd" are extremely sparse. Other meaningful features were created to replace such columns.
- Discount Percentage: Expresses the % difference between price_usd and sale_price_usd. Products with no discounts are labelled 0.
```python
df['discount_percentage'] = ((df['price_usd'] - df['sale_price_usd']) / df['price_usd']) * 100
df['discount_percentage'] = df['discount_percentage'].fillna(0)
```
- Relative Price Index: Divides individual product prices over average prices for each product’s corresponding secondary category.
```python
# Calculate the average price per category
avg_price_per_category = df.groupby("secondary_category")["price_usd"].mean()
# Merge the average category price back into the main dataframe
df["avg_category_price"] = df["secondary_category"].map(avg_price_per_category)
# Compute the Relative Price Index
df["relative_price_index"] = df["price_usd"] / df["avg_category_price"]
```

#### 2. Large Spread of Values
To minimise variances in values for some variables e.g. "loves_count", "reviews", "ratings", log transformation and binning of values were performed.

![image](https://github.com/user-attachments/assets/7b0e81fe-acb2-41f1-a8a7-1171499c1973)
![image](https://github.com/user-attachments/assets/e7e9ccac-3239-4fb2-b99d-48813adfc253)

**Log Transformation**
```python
df['log_loves'] = np.log1p(df['loves_count'])
df['log_reviews'] = np.log1p(df['reviews'])
```

**Binning**
```python
# Define rating bands (bins) and labels
rating_bins = [0, 1, 2, 3, 4, 5]
rating_labels = ["0", "1", "2", "3", "4"] # where label = "0" value = <1, where label = "4" value = 4-5
# Create a new column for rating bands
df["rating_band"] = pd.cut(df["rating"], bins=rating_bins, labels=rating_labels, include_lowest=True)
# Create log-transformed bins for "loves" and "reviews" based on 4 quantiles
df['log_loves_band'] = pd.qcut(df['log_loves'], q=4, labels=['1','2','3','4'])
df['log_reviews_band'] = pd.qcut(df['log_reviews'], q=4, labels=['1','2','3','4'])
```

#### 3. Highly Granular Values
Other variables e.g. "size", "variation_type", "variation_value" have a few thousand unique values. They generally describe the kinds of variations e.g. formulation, scent, colour that a product has.

**Summary Statistics of Some Categories:**
![image](https://github.com/user-attachments/assets/381eaf19-03cb-4d1c-b553-2ce6bc58d253)

For the model to better understand these variations, new features e.g. "variation_formulation", "variation_scent" were engineered to replace these variables.

```python
# Rename column 'variation_type' to 'variation_property' to prevent subsequent duplication
df.rename(columns={'variation_type': 'variation_property'}, inplace=True)

# Ensure 'variation_property' is a string before processing and replace null values with '0'
df['variation_property'] = df['variation_property'].astype(str).replace('nan', None)

# Create new columns to indicate the presence of different variation types, setting initial values to 0
# This step ensures that every product has a column for each variation type, even if they don’t apply
variation_columns = ['variation_color', 'variation_formulation', 'variation_scent', 'variation_size', 'variation_concentration', 'variation_type']
for col in variation_columns:
    df[col] = 0  # Start all variation flags at 0, and later update relevant rows to 1

# Assign 1 where applicable based on 'variation_property' values
df.loc[df['variation_property'].str.contains('Color', na=False), 'variation_color'] = 1
df.loc[df['variation_property'].str.contains('Formulation', na=False), 'variation_formulation'] = 1
df.loc[df['variation_property'].str.contains('Scent', na=False), 'variation_scent'] = 1
df.loc[df['variation_property'].str.contains('Size', na=False), 'variation_size'] = 1
df.loc[df['variation_property'].str.contains('Concentration', na=False), 'variation_concentration'] = 1
df.loc[df['variation_property'].str.contains('Type', na=False), 'variation_type'] = 1
```

For instance, if a product varies in formulation, size and concentration, it has value = 1 (i.e. yes) in the respective variables, and 0 (i.e. no) in other variation_XX columns.


![image](https://github.com/user-attachments/assets/eec737da-1e47-4457-aa3b-77368105e6a0)


**Other Data Cleaning Steps** 

Final data cleaning steps were performed below to prepare the dataset for modelling.

#### Rename Column
- Column "child_count" was renamed to "variation_count" to clearly identify the number of product variations.

```python
df.rename(columns={'child_count': 'variation_count'}, inplace=True)
```

#### Label Encoding Categorical Variables
```python
# Identify categorical columns that need encoding
categorical_cols = ["primary_category", "secondary_category", "tertiary_category", "rating_band","log_loves_band","log_reviews_band"]

# Apply Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
```

#### Dimension Reduction
- A correlation matrix was computed and "variation_concentration" was removed as it is highly correlated with "variation_formulation".

![image](https://github.com/user-attachments/assets/0f06858d-acc6-4a88-9050-add4bb3c21d1)

- As new features had been engineered, variables made redundant e.g. "loves_count", "reviews", "variation_value", " were also removed. Other variables of high cardinality, granularity e.g. "ingredients", "highlights", "size" that do not contribute meaningful patterns were also dropped.

```python
columns_to_remove = [
    "reviews",
    # Removed as log_reviews and log_reviews_band created to address skewed data
    "loves_count",
    # Removed as log_loves and log_loves_band created to address skewed data
    "product_id", 
    # Removed because it contains too many unique values, making it irrelevant for machine learning models
    "product_name", 
    # Removed for the same reason as 'product_id'— high uniqueness means it doesn’t contribute useful patterns
    "rating",
    # Removed as new "rating_band" column has been created to handle ratings
    "brand_name", 
    # Removed because the brand is already represented by 'brand_id', avoiding redundant information
    "size", 
    # Removed due to high cardinality (over 2,000 unique values), making it too varied for meaningful analysis
    "variation_property", 
    "variation_value",
    "variation_desc", 
    # Removed above 3 columns because new variation columns now handle product variations explicitly
    "ingredients", 
    # Removed as ingredient lists are highly specific to product, making it difficult to generalize for modeling
    "child_max_price",
    "child_min_price",
    "value_price_usd", 
    "sale_price_usd",
    # Removed above due to a high percentage of missing values, making them unreliable for analysis. More 
    # meaningful features like discount percentage / relative price index have also been created.
    "highlights" 
    # Removed as product attributes (e.g., 'Vegan', 'Matte Finish') are too granular and do not contribute 
    # significantly to structured analysis."
   
]

# Remove the specified columns
df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
```

### Modelling
#### 1. Technique
Decision Tree and Random Forest were chosen for the following reasons.

**Decision Trees (DT)**
- DT may help capture nonlinear relationships between variables better, especially when data has variance in values / skewed distribution.
- DT is easily interpretable as it offers a visual ‘flowchart’ of how variables influence product prices (target variable).

**Random Forest (RF)**
- Since RF averages the outcome of an ensemble of DTs, RF may provide more accurate predictions and alleviate issue of overfitting.
- RF is also interpretable by its feature importance metrics and identifies variables that are influential to product prices.

#### 2. Test Design and Construction
- In order to identify key factors driving product prices, a list of features was created and the target variable was defined.
  
![image](https://github.com/user-attachments/assets/17511b33-b9ae-4abd-b2ee-d7d0019bc9a2)

- Data was split into 80% training and 20% testing sets.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- In order to strike an optimal balance between under- and overfitting and to improve performance, respective hyperparameters e.g. tree depth were tuned in the DT and RF models. For each model, a 5-fold cross-validation function was used to derive the best combination of parameters. The models were then respectively trained on 80% training data and the best combination of hyperparameters.

**DT**

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid_dt = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Create the DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)

# GridSearchCV
grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    scoring='neg_mean_squared_error',  # or 'r2'
    cv=5,
    n_jobs=-1
)

# Fit on training data
grid_dt.fit(X_train, y_train)

# Best estimator and parameters
best_dt = grid_dt.best_estimator_
```

```python
print("Best Decision Tree Parameters:", grid_dt.best_params_)
Best Decision Tree Parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5}
```

**RF**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Define parameter distributions for RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

rf = RandomForestRegressor(random_state=42)

random_rf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist_rf,
    scoring='neg_mean_squared_error',  # or 'r2'
    n_iter=20,  
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_rf.fit(X_train, y_train)
best_rf = random_rf.best_estimator_
```

```python
print("Best Random Forest Parameters:", random_rf.best_params_)
Best Random Forest Parameters: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 10}
```

### Evaluation
#### 1. Criterion
The models were evaluated using the same set of criteria:
- Root Mean Squared Error (RMSE): Measures accuracy of price predictions by predicting how far off the predicted price is from actual price. **Reliable prices can be forecasted to aid pricing strategies.**
- Coefficient of Determination (R²): The higher the R², the better the model is at explaining why product prices vary based on features included. **This provides confidence that key variables impact pricing decisions.**

#### 2. Model Comparison: 
Based on evaluation criterion, RF is the more superior model.

![image](https://github.com/user-attachments/assets/08c71f98-8325-40a7-ae1b-1e6830fa5dc3)

- While both models have relatively high RMSE, the RF model has lower RMSE than DT model. This shows that the RF model is better at making price predictions as predictions are much closer to actual prices, and can thereby better aid pricing decisions.
- Compared to DT, RF has higher R². While DT model captures 35% of variation in product prices, RF model captures 81% of variation in product prices. This means that RF model captures most of the factors that influence product prices, which is critical to making informed pricing decisions.

**RF: Feature Importance Metrics**

In an especially competitive online market, feature importance metrics in RF model could be useful in precisely identifying features that drive product prices.

![image](https://github.com/user-attachments/assets/2450f4fe-0568-4848-a588-a16cba035951)

- Sephora’s product pricing is heavily influenced by how product prices compare with the average prices in respective product category and average product category prices as they contribute to more than 98% of the model's total predictive power.
- Other categorical hierarchies (e.g. tertiary_category) also contribute albeit less significantly.
- Remaining features (e.g. limited_edition, sephora_exclusive) may not drive pricing predictions, but instead influence customers’ perception, purchasing habits etc.


**RF: Partial Dependence Plots of Highly Important Features**

Inclining curve in Partial Dependence Plots show how higher relative_price_index or avg_category_price corresponds to higher predicted price (with all other variables held constant).

![image](https://github.com/user-attachments/assets/d1553927-d1ff-43cf-b42e-685c94c4ba29)

- For instance, when relative price index = 1.0, product price is predicted at about $50. When average category price is about $60, product price is predicted to be between $60 to $70.

## Recommendation and Analysis
Dominant drivers like relative_price_index and avg_category_price, coupled with contributions (albeit smaller) from features like tertiary_category, secondary_category reinforces that **understanding product types / categories / classification is critical** for the model to predict pricing and hence inform pricing decisions. 

The following recommendations may work hand-in-hand to strengthen this understanding and strategise pricing in order to capture online market share:
1. Category Price Tracking System: Product prices are monitored for shifts and trends within their respective tertiary, secondary and primary categories. 
2. Categorical Price Positioning: For price-sensitive consumers interested in particular products from a specific category e.g. lip gloss (tertiary category), use price predictions to appropriately price new products in relation to respective categorical prices.
3. Dynamic Pricing based on Categorical Shifts: As brands may set own baseline pricing standards for their products, and which may change in response to non-price sensitive factors like brand popularity, dynamically update prices of products in existing catalog depending on how prices of products within respective category shifts.  
4. As competitive landscape evolves, model should be regularly retrained (in consideration of business issues specified below)

**Business Issues**

From a price prediction perspective, many non-price attributes e.g. ratings_band, brand_id (i.e. brand name) appear to contribute minimally. Data available may not have enough variations or are unmeaningful.
1. Enhance data collection. More conclusive indicators of brand value e.g. marketing spend by brand/brand product may help predict how brand value influences products’ prices.
2. Additionally, expand data collection beyond brand-related or product-related attributes to explain price variations. For instance, 
- As consumer demographic data (e.g. age, income level, geographical location) may inform spending habits and price variations based on demographic or region, consumer-centric features may be engineered to help model evolve its predictive power and position prices.
- To enhance the impact of price-related attributes like discount_percentage, time-series data like season/month of promotion, promotion duration can be collected and related features engineered.
3. The current model involves data on all products across categories. The outcome suggests that prices are category-dependent and -specific. Models specifically trained on particular categories e.g. skincare could be more useful for product pricing within respective category.

With enhanced data collection, data preparation and model design, model is prepped to make more accurate pricing predictions. This is essential to help SEPHORA inform pricing decisions and implement strategies as it reacts to an evolving market and sensitises to consumer needs. Such is ultimately important for SEPHORA to increase sales and market share.

## AI Ethics

**Privacy**

If consumer-centric data e.g. demographic-related like age, gender, skin texture, income or browsing habits like products visited are integrated to enhance the pricing model, data must be captured and handled such that consumers' privacy is safeguarded. Consumers must be informed about data collection and consent to potential uses and applications. Sensitive consumer identifiers must also be anonymised to enhance data privacy. Without which, data privacy is at risk and consumers' trust could be eroded.


**Fairness**

Some existing variables e.g. certain brands and their related products are underrepresented in the dataset. Colour-specific product features like **colour tone** or **colour shade** are not encapsulated either. Hence, minority brands and skin tone/colour might be undervalued and model could result in bias as such groups are disadvantaged in representation. Conversely, brands and products with more data may be overrepresented. The model might make erroneous predictions in prices for implicated products, and reinforce certain cultural or economic stigmas e.g. socio-economic disparity.


**Accuracy**

When certain variables are deemed to be insignificant to the model predictions and are omitted, the model may result in inaccurate predictions and misguide pricing decisions and strategies. For instance, without time-series data e.g. season/month of promotion, recommended prices may fail to reflect seasonal pricing trends. Without competitive product prices, Sephora may lose its appeal and consumer base. For instances where data collection is biased, inaccurate and/or incomplete, the model may predict prices that favour the oversampled majority class and fail to predict those that are emerging.


**Accountability**

If variables are omitted or certain classes favoured due to any of the above reasons e.g. biased data collection, price predictions may be inaccurate or unstable. Decision makers and relevant stakeholders may find it challenging to explain and justify pricing strategies that result from the model's predictions. Should products be mispriced due to this model, the trajectory for redressing pricing decisions could also be unclear.


**Transparency**

The Random Forest model may be too intricate for a non-technically trained stakeholder to understand its construct. Without clear layman explanations, stakeholders may find it challenging to understand, trust, and rely on the model for the business' pricing decisions. The processes of data collection e.g. data sources, collection period, data preparation e.g. feature selection, dimension reduction, model construction, and model evaluation e.g. criterion must also be clearly communicated and understood. This would reduce any missteps e.g. poor data integrity, model bias that could challenge the model's outcomes and reliability.  Stakeholders are consequently primed to be in a better position to excavate areas for data and/or model improvement and adjust pricing strategies.


## Source Codes and Datasets
Link: [https://github.com/chingchriselle/ITD214-Workbook]

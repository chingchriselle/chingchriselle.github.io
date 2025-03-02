layout: post
author: Ching Chriselle
title: "Applied Data Science Project"
categories: ITD214
---
## Project Background
### Industry Background:
The Beauty and Personal Care industry is projected to exceed US$100bil in revenues in 2025. In the next 5 years, annual industry growth is estimated to hit 2.9% [[source](https://www.statista.com/outlook/cmo/beauty-personal-care/united-states)]. More importantly, sales from online avenues are expected to account for about 65.9% of 2025 revenues. SEPHORA is a leading beauty and personal care products retailer. It offers a catalogue of brand and housebrand goods across brick-and-mortar stores as wellas online. In 2022, its revenues from the United States reached almost US$7bil, and accounted for more than half of its global revenues [[source](https://www.statista.com/statistics/1445562/retail-sales-of-sephora-globally/)].

### Business Goal:
To improve online channel sales at Sephora

### Objectives:
1. To identify key factors driving product prices
2. To predict customer sentiment towards beauty products based on review text

## Work Accomplished
This project explores how objective 1 can be achieved to meet Sephora's business goal. 

### Data Source:
Information on products and customer reviews on its skincare catalogue were scrapped from SEPHORA’s online store (collected as of Mar 2023) and published in KAGGLE [[source](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/)]. Dataset:
- Consists of 6 csv data files for over 8,000 products and more than 1 million customer reviews.
- Includes 1 file “product_info” that provides product information and 5 files “reviews...” that provide customer reviews on skincare products. Both groups of files can be joined by a unique assigned product identifier “product_id”.
- Includes information like product details, prices, ratings, ingredients, user free-text reviews, etc.

### Data Preparation
Objective 1 uses only data from "product_info". The data file contains 27 columns and over 8,000 rows of numerical and categorical values. Each row points to each distinct product that can be identified by its "product_id". 

#### Dataframe:
![image](https://github.com/user-attachments/assets/bd11a90b-4ed5-428b-aaf8-7a6c5633fd25)

Upon data exploration, data cleaning was applied to 3 key problem areas outlined below.

#### 1. Sparse Columns with Null Values
Several variables had many null values. They were either imputed by logic, or new features were engineered to replace such columns.
![image](https://github.com/user-attachments/assets/4399cab3-823c-4b93-8d2a-d5b332c08d11)

#### Imputation by Logic
- Imputed null values in "tertiary_category" as “Unknown” to demonstrate that these products do not have such in-depth categorical differentiation.
```
df['tertiary_category'] = df['tertiary_category'].fillna('Unknown')

```
- There are 8 empty values in "secondary_category". They were imputed variedly below.
```
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
```
df['rating'] = df['rating'].fillna(0)
df['reviews'] = df['reviews'].fillna(0)
```
#### Feature Engineering
Some price-related variables e.g. "sale_price_usd", "value_price_usd" are extremely sparse. Other meaningful features were created to replace such columns.
- Discount Percentage: Expresses the % difference between price_usd and sale_price_usd. Products with no discounts are labelled 0.
```
df['discount_percentage'] = ((df['price_usd'] - df['sale_price_usd']) / df['price_usd']) * 100
df['discount_percentage'] = df['discount_percentage'].fillna(0)
```
- Relative Price Index: Divides individual product prices over average prices for each product’s corresponding secondary category.
```
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

#### Log Transformation
```
df['log_loves'] = np.log1p(df['loves_count'])
df['log_reviews'] = np.log1p(df['reviews'])
```
#### Binning
```
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

#### Summary Statistics of Some Categories:
![image](https://github.com/user-attachments/assets/381eaf19-03cb-4d1c-b553-2ce6bc58d253)

For the model to better understand these variations, new features e.g. "variation_formulation", "variation_scent" were engineered to replace these variables.
```
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

### Modelling
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

### Evaluation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Recommendation and Analysis
Explain the analysis and recommendations

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Source Codes and Datasets
Link to GitHub repo: https://github.com/chingchriselle/ITD214-Workbook

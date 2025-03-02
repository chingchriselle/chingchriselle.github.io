![image](https://github.com/user-attachments/assets/bcddd132-a33f-4d1f-b1d2-6b47963e4121)---
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

Upon data exploration, data cleaning was applied to 3 key areas outlined below.

##### 1. Sparse Columns with Null Values
Several columns had many null values. They were either imputed by logic, or new features were engineered to replace such columns.
![image](https://github.com/user-attachments/assets/4399cab3-823c-4b93-8d2a-d5b332c08d11)

##### Imputation by Logic
- Imputed null values in tertiary_category as “Unknown” as product types likely do not require such in-depth categorical differentiation.
```python
df['tertiary_category'] = df['tertiary_category'].fillna('Unknown')
df['tertiary_category'].isnull().sum()
```
- Zeroised null ratings and reviews i.e. products have not ever been rated or reviewed.
```python
df['rating'] = df['rating'].fillna(0)
df['reviews'] = df['reviews'].fillna(0)
```
##### Feature Engineering
Discount Percentage | Expresses the % difference between price_usd and sale_price_usd. Products with no discounts = 0
Relative Price Index | Divides individual product prices over average prices for each product’s corresponding secondary category

#### 2. Large Spread of Values

#### 3. Highly Granular Values
   
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

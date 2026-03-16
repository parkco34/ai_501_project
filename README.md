# Online Shoppers Purchasing Intention Prediction
Course: AAI-501 – Foundations of Artificial Intelligence
Institution: University of San Diego
Team Members: Nathan Butcher, Paola Marsal, Cory Parker, Glen Salazar
Date: March 2026

# Project Overview
This project investigates the prediction of *online purchasing intention* using machine learning methods applied to behavorial session data from an e-commerce platform.

The dataset used is the *Online Shoppers Purchasing Intention Dataset* from the *UCL Machine Learning Repository* (Sakar & Kastro, 2018).  The dataset contains *12,330 user sessions collected over a year*, where each record represents a single visitor interaction with an online shopping website.  

Each session contains *10 numerical features and 8 categorical features* describing user behavior which includes:
    - Number of pages visited
    - Duration of visits
    - Bounce rate (percentage of visitors who land on website and leave without interacting with the site)
    - Exit rate
    - Page value
    - Traffic Source
    - Visitor type
    - Returning visitor status

*Binary Classfication Problem*
The target variable is *Revenue ∈ {0,1}*, where
    0 = Session didn't result in a purchse
    1 = Session did result in a purchase

## Business Motivation
Predicting purchase intention is important in for e-commerce platforms.
Accurate prediction enables organizations to:
    - Personalize user experiences in real time
    - Optimize marketing and advertising spend
    - Reduce rate which the cart is abandoned
    - Target high-value customers
    - Improve conversion rates
   
## Algorithms Used
1. __K-Means Clustering (Unsupervised Learning)__
K-Means will be used to locate **behavioral segments of shoppers** based on browsing patterns.

2. __Logistic Regression__
Logistic Regression provides an *interpretable baseline classifier*.
The model estimates the probability of a purchase:
> $$P(y = 1|x) = \frac{1}{1 + e^{-(\beta_{0} + \beta^{T}x)}}$$


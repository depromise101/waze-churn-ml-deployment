# Project Overview
This project demonstrates an end-to-end machine learning deployment for predicting user churn in the Waze app. The model is trained using historical user behavior data and deployed as a live web application using Hugging Face and Render.

The deployed model was validated by testing multiple user scenarios through the live interface, producing consistent and logically aligned churn predictions, confirming that the system performs real-time inference using the trained XGBoost model.

# Business Problem
This project extends the original churn analysis by deploying the trained model into a live production environment using Hugging Face and Render, allowing non-technical users to input user features and receive real-time predictions via a web interface.

# Model Used
Both the Random Forest Classifier and XGBoost Classifier were evaluated using cross-validation. While the Random Forest model offered strong baseline performance and stability, the XGBoost model demonstrated better predictive accuracy and was therefore chosen as the final model for deployment.

# How It Works
The system trains two machine learning models: Random Forest and XGBoost, using historical user behavior data. Both models are evaluated using a technique called cross-validation, which tests how well each model performs on different subsets of the data to ensure reliable results.

The Random Forest model serves as a strong and stable baseline, providing consistent predictions and helping validate the overall modeling approach. The XGBoost model achieves higher predictive accuracy, meaning it is better at correctly identifying users who are likely to churn.

Based on this performance comparison, XGBoost is selected as the final model and deployed in the application to generate real-time churn predictions for users.

# Screenshots
The following screenshots demonstrate the deployed XGBoost model performing real-time churn predictions. Different user profiles produce different outputs, validating that the model is actively processing inputs and not returning static results.

# Live Link
https://huggingface.co/spaces/ASAPDELIVERY/waze-churn-predictor

https://automated-azure-project.onrender.com










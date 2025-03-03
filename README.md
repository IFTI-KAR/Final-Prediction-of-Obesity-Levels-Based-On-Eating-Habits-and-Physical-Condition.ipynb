# Prediction of Obesity Levels Based on Eating Habits and Physical Condition

## Overview
This project aims to predict obesity levels using machine learning techniques based on individuals' eating habits and physical conditions. The dataset includes various lifestyle and health-related attributes, and the model provides predictions on whether a person is underweight, normal weight, overweight, or obese.

## Dataset
The dataset contains features related to:
- Eating habits
- Physical activity
- Health conditions
- Demographic factors

## Project Structure
- `Prediction_of_Obesity_Levels_Based_On_Eating_Habits_and_Physical_Condition_(ML_project).ipynb`: Jupyter Notebook containing data preprocessing, exploratory data analysis (EDA), model training, and evaluation.
- `README.md`: Documentation for the project.
- `image.png`: A bar chart comparing the precision and recall scores of different machine learning models (KNN, Logistic Regression, Decision Tree, and Naïve Bayes).

## Steps Involved
1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and normalizing data.
2. **Exploratory Data Analysis (EDA)**: Visualizing distributions, correlations, and feature importance.
3. **Model Training**: Implementing machine learning models such as Logistic Regression, Decision Tree, Random Forest, and others.
4. **Evaluation**: Assessing model performance using accuracy, precision, recall, and F1-score.

## Model Performance Visualization
The project includes a precision and recall comparison for different models. The image below represents the comparison:

![Precision and Recall Comparison](image.png)

### Key Observations from the Chart
- **Decision Tree** performs the best among all models, achieving nearly perfect precision and recall scores (~0.99).
- **KNN** and **Naïve Bayes** exhibit similar performance, with precision and recall around 0.80.
- **Logistic Regression** has slightly lower precision and recall (~0.75), making it the least effective model in this comparison.
- The **small performance gap between precision and recall** for each model suggests a well-balanced classification without major bias towards false positives or false negatives.

## Dependencies
Ensure you have the following dependencies installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run the Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/IFTI-KAR/Final-Prediction-of-Obesity-Levels-Based-On-Eating-Habits-and-Physical-Condition.ipynb/tree/main
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Run the notebook step by step to preprocess data, train models, and analyze results.

## Results
The trained model provides predictions on obesity levels based on input features. Model performance is evaluated using appropriate metrics, and Decision Tree is identified as the best-performing model based on precision and recall scores.

## Future Improvements
- Experimenting with deep learning techniques.
- Incorporating additional health-related features.
- Deploying the model as a web application.

## License
This project is open-source and available under the MIT License.


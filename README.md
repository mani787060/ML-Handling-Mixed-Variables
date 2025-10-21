# Handling Mixed Variables in Machine Learning

-> This project demonstrates how to handle datasets that contain both **numerical and categorical variables**, a common scenario in real-world ML problems.  
-> By applying suitable preprocessing techniques to each type of feature, we ensure that machine learning models interpret data correctly and perform efficiently.


## Topics Covered:-
-> Identifying numerical and categorical features  
-> Scaling numerical variables  
-> Encoding categorical variables  
-> Combining transformed features  
-> Preparing the final dataset for modeling  

## Libraries Used:-
-> pandas  
-> numpy  
-> scikit-learn  

## Techniques Used:-
-> **StandardScaler()** for numerical columns  
-> **OneHotEncoder() / OrdinalEncoder()** for categorical columns  
-> **ColumnTransformer** for combined preprocessing  

## Key Takeaway:-
Properly handling mixed variables ensures:
-> Better model performance  
-> More consistent feature scaling  
-> Easier integration into pipelines and ML workflows  

**Note:**  
-> This notebook uses the **Titanic dataset** to demonstrate how to preprocess numeric (e.g., `Age`, `Fare`) and categorical (e.g., `Sex`, `Embarked`) features together.

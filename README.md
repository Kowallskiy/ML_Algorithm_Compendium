# Machine Learning
___
__Language__: English, [Русский](rus.md)

## Contents

* [Algorithms](Contents/algorithms_eng.md)
* [Classification](Contents/Classification_eng.md)
* [Clustering](Contents/clustering_eng.md)
* [Theory](Contents/theory.md)
* [Scaling](Contents/scaler_eng.md)

## Linear Regression

### Preprocessing
```Python
# Converting the 'Date' feature to a date and time format
melb_data["Date"] = pd.to_datetime(melb_data["Date"], format="%d/%m/%Y")
# The 'Date' feature is displaying only the year
melb_data["Date"] = melb_data["Date"].dt.year
```

To further process the data, it is necessary to identify _missing values_:

|Feature|№|
|---|---|
|car|62|
|BuildingArea|6450|
|YearBuilt|5375|
|CouncilArea|1369|

Next, we determine the number of unique values for categorical features:
```Python
melb_data.select_dtypes(['object']).nunique()
```

|Feature|№|
|---|---|
|Suburb|314|
|Address|13378|
|Type|3|
|Method|5|
|SellerG|268|
|CouncilArea|33|
|regionname|8|

Next, let's consider three different approaches to model training:

### Approach 1: Deleting rows with missing values

```Python
# Drop rows with missing values
melb_data.dropna(axis=0, inplace=True)

# Eliminate certain features from the feature matrix
X = melb_data.drop(["Price", "Suburb", "Address", "SellerG", "CouncilArea"], axis=1)
# Target variable vector
y = melb_data["Price"]

print(X.shape)
```

> (6196, 16)

```Python
# Separate the feature matrix into categorical and numerical
categorical_val = ["Type", "Method", "Regionname"]
numerical_val = X.columns.difference(categorical_val)

# Create a pipeline to transform categorical variables into binary representations
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Apply this pipeline to categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_val),
        ('cat', categorical_transformer, categorical_val)
    ]
)

# Combine preprocessing and machine learning model into a pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Split the dataset into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
 
# Train the model and predict target variable values
pred = pipe.fit(train_X, train_y).predict(test_X)
```
```Python
# Use MAPE metric for accuracy evaluation
absolute_percentage_error = np.abs((test_y - pred) / test_y) * 100
mape = np.mean(absolute_percentage_error)
```
> MAPE: __27.934158203951704__

### Approach 2: Machine Learning Method for Handling Missing Values

To avoid reducing the dataset by half, we'll use a machine learning method to predict the values of the __BuildingArea__ and __YearBuilt__ features.

```Python
# Selecting features for prediction
relevant_features = ["Rooms", "Type", "Bedroom2", "Distance", "Postcode", "Bathroom", "Car", "Landsize", "Regionname"]
# Filtering the rows with non-null BuildingArea values for training
X_train = melb_data.loc[~melb_data["BuildingArea"].isnull(), relevant_features]
y_train = melb_data.loc[~melb_data["BuildingArea"].isnull(), "BuildingArea"]
# Selecting rows of BuildingArea where values are null
X_missing = melb_data.loc[melb_data["BuildingArea"].isnull(), relevant_features]
# Creating a pipeline with a model and OneHotEncoder for categorical variables
pipe = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('model', LinearRegression())
])
# Training the model and predicting missing values
y_missing_pred = pipe.fit(X_train, y_train).predict(X_missing)

# Filling missing BuildingArea values with predicted values
melb_data.loc[melb_data["BuildingArea"].isnull(), "BuildingArea"] = y_missing_pred
```
Applying the same algorithm to predict __YearBuilt__:

```Python
relevant_features = ["Rooms", "Method", "Distance", "Postcode", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "Regionname"]

X_train = melb_data.loc[~melb_data["YearBuilt"].isnull(), relevant_features]
y_train = melb_data.loc[~melb_data["YearBuilt"].isnull(), "YearBuilt"]

X_missing = melb_data.loc[melb_data["YearBuilt"].isnull(), relevant_features]

pipe = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('model', LinearRegression())
])

y_missing_pred = pipe.fit(X_train, y_train).predict(X_missing)

melb_data.loc[melb_data["YearBuilt"].isnull(), "YearBuilt"] = y_missing_pred
```
```Python
# Dropping rows with missing 'Car' values (total of 62)
melb_data = melb_data.dropna(subset=["Car"])

X = melb_data.drop(["Price", "Suburb", "Address", "SellerG", "CouncilArea"], axis=1)
y = melb_data["Price"]

print(X.shape)
```

> (13518, 16)

```Python
# Applying the same approach as in the first case
categorical_cols = ["Type", "Method", "Regionname"]
numerical_cols = X.columns.difference(categorical_cols)

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pred = pipe.fit(X_train, y_train).predict(X_test)

# Using Ridge Regression (L2 regularization method)
pipe_ridge = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])

pipe_ridge.fit(X_train, y_train)
y_pred = pipe_ridge.predict(X_test)
```
```Python
absolute_percentage_error = np.abs((y_test - pred) / y_test) * 100
mape = np.mean(absolute_percentage_error)
```
MAPE: __27.67606068694708__

## Approach 3: Polynomial Features

In this example, I decided to enhance the model to account for the nonlinear features in the input data.

This method was also used by me in  [scaling](Scaler.md)

```Python
# In this example, we fill missing values with zeros in the "Car" column
X["Car"] = X["Car"].fillna(0)

categorical_cols = ["Type", "Method"]
numerical_cols = X.columns.difference(categorical_cols)
# Apply OneHotEncoder to categorical variables to transform them into binary values
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Fill missing values with mean values for numerical features
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])
# Apply pipelines to the variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])


# Apply the specified preprocessor and PolynomialFeatures for learning nonlinear relationships
# In this case, degree=2 is used to avoid overfitting
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('poly_features', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
```
```Python
absolute_percentage_error = np.abs((y_test - pred) / y_test) * 100

mape = np.mean(absolute_percentage_error)
```
MAPE: __21.96483206600583__


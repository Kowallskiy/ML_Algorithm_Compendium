# Дерево решений (Decision Trees)
Preprocessing:
```Python
X = heart_data.drop(["Height_(cm)", "Weight_(kg)", "Heart_Disease"], axis=1)
y = heart_data["Heart_Disease"]

categorical_columns = ["General_Health", "Checkup", "Exercise", "Skin_Cancer",
                       "Other_Cancer", "Depression", "Diabetes", "Arthritis", "Sex", "Age_Category", "Smoking_History"]
numerical_columns = X.columns.difference(categorical_columns)
```
Pipelines and prediction:
```Python
def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):

    preprocessor = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', 'passthrough', numerical_columns)
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0))
    ])

    pred = pipe.fit(X_train, y_train).predict(X_test)
    return pred

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.9)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    accuracy = accuracy_score(y_test, my_mae)
    print(f"Accuracy: {accuracy:.6f}")
```
___Accuracy__: _0.919316_
___Accuracy__: _0.918992_
___Accuracy__: _0.916532_
___Accuracy__: _0.907175_

# Случайный лес (Random forest)
```Python
from sklearn.ensemble import RandomForestClassifier

preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numerical_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=0))
])

y_pred = pipeline.fit(X_train, y_train).predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred)
print(f"Random Fores accuracy: {rf_accuracy: .6f}")
```
__Accuracy__: _0.915949_
# Градиентный бустинг (Gradient Boosting)
```Python
from sklearn.ensemble import GradientBoostingClassifier

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=0))
])

y_pred = pipeline.fit(X_train, y_train).predict(X_test)

gb_accuracy = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting accuracy: {gb_accuracy:.6f}")
```
__Accuracy__: _0.918636_
# K-ближайших соседей (K-neighbors classifier)
```Python
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numerical_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
])

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', PolynomialFeatures()),
    ('model', KNeighborsClassifier(n_neighbors=10))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.9)

y_pred = pipe.fit(X_train, y_train).predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.6f}")
```
__Accuracy__: _0.918962_

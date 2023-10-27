# Методы машинного обучения

## Содердание

* Алгоритмы
* Классификация
* Кластеризация
* [Теория (eng)](theory.md)
* [Масштабирование](Scaler.md)

## Линейная регрессия (Linear Regression)

### Предварительная оработка данных (Preprocessing)
```Python
# Конвертация формата признака Date в формат даты и времени
melb_data["Date"] = pd.to_datetime(melb_data["Date"], format="%d/%m/%Y")
# Признак Date отображает только год
melb_data["Date"] = melb_data["Date"].dt.year
```

Для последующей обработки данных необходимо выявить _пропущенные значения_:

|car|62|
|___|___|
|BuildingArea|6450|
|YearBuilt|5375|
|CouncilArea|1369|

Затем узнаём количество уникальных значений категориальных признаков:
```Python
melb_data.select_dtypes(['object']).nunique()
```

|Suburb|314|
|Address|13378|
|Type|3|
|Method|5|
|SellerG|268|
|CouncilArea|33|
|regionname|8|

Далее рассмотрим разных подхода к обучению модели:

### Удаление строк с отсутсвующими значениями

```Python
# Удаляем строки с отсутсвующими значениями
melb_data.dropna(axis=0, inplace=True)

# Устраняем следующие признаки из матрицы признаков
X = melb_data.drop(["Price", "Suburb", "Address", "SellerG", "CouncilArea"], axis=1)
# Вектор целевой переменной
y = melb_data["Price"]

print(X.shape)
```

> (6196, 16)

```Python
# Разделяем матрицу признаков на категориальные и номинальные
categorical_val = ["Type", "Method", "Regionname"]
numerical_val = X.columns.difference(categorical_val)

# Создаём конвеер для преобразования категориальных переменных в бинарные представления
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Применяем данный конвеер на категориальные переменные
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_val),
        ('cat', categorical_transformer, categorical_val)
    ]
)

# Объединяем предварительную обработку и модель машинного обучения в пайплайн
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Разделяем набор данных на обучающую и тестовую выборки
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
 
 # Обучаем модель предсказываем значения целевой переменной
pred = pipe.fit(train_X, train_y).predict(test_X)
```
```Python
# Используем метрику MAPE для оценки точности
absolute_percentage_error = np.abs((test_y - pred) / test_y) * 100
mape = np.mean(absolute_percentage_error)
```
> MAPE: __27.934158203951704__

### Метод машинного обучения для заполнения отсутсвующих значений

Чтобы не уменьшать количество данных в 2 раза, используем метод машинного обучения для предсказания значений признаков __BuildingArea__ и __YearBuilt__.

```Python
# Выбираем признаки для предсказания
relevant_features = ["Rooms", "Type", "Bedroom2", "Distance", "Postcode", "Bathroom", "Car", "Landsize", "Regionname"]
# Отбираем строки выбранных признаков, где значения BuildingArea не пропущены
X_train = melb_data.loc[~melb_data["BuildingArea"].isnull(), relevant_features]
# Выбираем строки признака BuildingArea, где значения не пропущены
y_train = melb_data.loc[~melb_data["BuildingArea"].isnull(), "BuildingArea"]
# Отбираем строки выбранных признаков, где значения BuildingArea пропущены
X_missing = melb_data.loc[melb_data["BuildingArea"].isnull(), relevant_features]
# Создаём пайплайн с моделью и OneHotEncoder для категориальных переменных
pipe = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('model', LinearRegression())
])
# Обучаем модель и предсказываем пропущенные значения
y_missing_pred = pipe.fit(X_train, y_train).predict(X_missing)

# Заполняем пропущенные значения BuildingArea предсказанными значениями
melb_data.loc[melb_data["BuildingArea"].isnull(), "BuildingArea"] = y_missing_pred
```
Применяем тот же алгоритм для предсказания __YearBuilt__:

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
# Сбрасываем строки с пропущенными значенями признака Car (их всего 62)
melb_data = melb_data.dropna(subset=["Car"])

X = melb_data.drop(["Price", "Suburb", "Address", "SellerG", "CouncilArea"], axis=1)
y = melb_data["Price"]

print(X.shape)
```

> (13518, 16)

```Python
# Далее применяем тот же подход, как в первом случае
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

# Применяем метод Ridge (метод наименьших квадратов с L2 регуляризацией)
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

## Созданием полиномиальных признаков (Polynomial Features)

В данном примере я решил усложнить модель для учёта нелинейных особенностей входных данных.

Данный метод также использовался мной при [масштабировании](Scaler.md)

```Python
# В данном примере заполним пропущенные значения нулями
X["Car"] = X["Car"].fillna(0)

categorical_cols = ["Type", "Method"]
numerical_cols = X.columns.difference(categorical_cols)
# Применяем OneHotEncoder на категориальные переменные для преобразования их в бинарные значения
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Заполняем пропущенные значения среднимим значениями признаков
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])
# Применяем пайплайны на переменные
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

# Применяем обозначенный preprocessor и PolynomialFeatures для обучения на нелинейных закономерностях
# В данном случае с degree=2, чтобы не спровоцировать overfitting
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


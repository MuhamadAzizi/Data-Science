# Machine Learning

Machine Learning adalah suatu bidang ilmu yang memungkinkan mesin untuk mempelajari pola-pola berdasarkan data.

Belajar Machine Learning berdasarkan dataset yang diambil dari `kaggle` yaitu `Melbourne Housing Snapshot`.

Machine Learning ini digunakan untuk memprediksi `price` berdasarkan atas pertimbangan-pertimbangan(`Features`) yang digunakan dalam proses Machine Learning.

## Import Dataset

Gunakan library `pandas` untuk meng-import dataset yang ada dan akan membuat dataset tersebut menjadi DataFrame.

`read_csv()` adalah fungsi/method pada library `pandas` yang digunakan untuk meng-import suatu dataset.


```python
import pandas as pd
```


```python
df = pd.read_csv('../../../datasets/melb_data.csv/melb_data.csv')
```

## Eksplorasi Dataset

Hal ini dilakukan untuk mengetahui karakteristik dari dataset yang akan kita gunakan dan untuk mengetahui juga apakah dalam dataset tersebut ada missing value atau tidak.

### Mencari Sampel

- Untuk mengetahui lima baris data pertama, bisa digunakan method `head()`
- Untuk mengetahui lima baris data terakhir, bisa digunakan method `tail()`


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Suburb</th>
      <th>Address</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Price</th>
      <th>Method</th>
      <th>SellerG</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>...</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>YearBuilt</th>
      <th>CouncilArea</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Regionname</th>
      <th>Propertycount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abbotsford</td>
      <td>85 Turner St</td>
      <td>2</td>
      <td>h</td>
      <td>1480000.0</td>
      <td>S</td>
      <td>Biggin</td>
      <td>3/12/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>202.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra</td>
      <td>-37.7996</td>
      <td>144.9984</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Abbotsford</td>
      <td>25 Bloomburg St</td>
      <td>2</td>
      <td>h</td>
      <td>1035000.0</td>
      <td>S</td>
      <td>Biggin</td>
      <td>4/02/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>79.0</td>
      <td>1900.0</td>
      <td>Yarra</td>
      <td>-37.8079</td>
      <td>144.9934</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Abbotsford</td>
      <td>5 Charles St</td>
      <td>3</td>
      <td>h</td>
      <td>1465000.0</td>
      <td>SP</td>
      <td>Biggin</td>
      <td>4/03/2017</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>150.0</td>
      <td>1900.0</td>
      <td>Yarra</td>
      <td>-37.8093</td>
      <td>144.9944</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Abbotsford</td>
      <td>40 Federation La</td>
      <td>3</td>
      <td>h</td>
      <td>850000.0</td>
      <td>PI</td>
      <td>Biggin</td>
      <td>4/03/2017</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>94.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra</td>
      <td>-37.7969</td>
      <td>144.9969</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Abbotsford</td>
      <td>55a Park St</td>
      <td>4</td>
      <td>h</td>
      <td>1600000.0</td>
      <td>VB</td>
      <td>Nelson</td>
      <td>4/06/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>142.0</td>
      <td>2014.0</td>
      <td>Yarra</td>
      <td>-37.8072</td>
      <td>144.9941</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### Mengetahui Baris dan Kolom

- `shape` digunakan untuk mengetahui baris dan kolom
- `columns` digunakan untuk mengetahui nama dari setiap kolom


```python
df.shape
```




    (13580, 21)




```python
df.columns
```




    Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
           'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
           'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
           'Longtitude', 'Regionname', 'Propertycount'],
          dtype='object')



### Melihat Ringkasan Dataset

Method `describe()` akan membuat ringkasan sederhana dari suatu dataset


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rooms</th>
      <th>Price</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>YearBuilt</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Propertycount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>13580.000000</td>
      <td>1.358000e+04</td>
      <td>13580.000000</td>
      <td>13580.000000</td>
      <td>13580.000000</td>
      <td>13580.000000</td>
      <td>13518.000000</td>
      <td>13580.000000</td>
      <td>7130.000000</td>
      <td>8205.000000</td>
      <td>13580.000000</td>
      <td>13580.000000</td>
      <td>13580.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>2.937997</td>
      <td>1.075684e+06</td>
      <td>10.137776</td>
      <td>3105.301915</td>
      <td>2.914728</td>
      <td>1.534242</td>
      <td>1.610075</td>
      <td>558.416127</td>
      <td>151.967650</td>
      <td>1964.684217</td>
      <td>-37.809203</td>
      <td>144.995216</td>
      <td>7454.417378</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.955748</td>
      <td>6.393107e+05</td>
      <td>5.868725</td>
      <td>90.676964</td>
      <td>0.965921</td>
      <td>0.691712</td>
      <td>0.962634</td>
      <td>3990.669241</td>
      <td>541.014538</td>
      <td>37.273762</td>
      <td>0.079260</td>
      <td>0.103916</td>
      <td>4378.581772</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>8.500000e+04</td>
      <td>0.000000</td>
      <td>3000.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1196.000000</td>
      <td>-38.182550</td>
      <td>144.431810</td>
      <td>249.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>2.000000</td>
      <td>6.500000e+05</td>
      <td>6.100000</td>
      <td>3044.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>177.000000</td>
      <td>93.000000</td>
      <td>1940.000000</td>
      <td>-37.856822</td>
      <td>144.929600</td>
      <td>4380.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>3.000000</td>
      <td>9.030000e+05</td>
      <td>9.200000</td>
      <td>3084.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>440.000000</td>
      <td>126.000000</td>
      <td>1970.000000</td>
      <td>-37.802355</td>
      <td>145.000100</td>
      <td>6555.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>3.000000</td>
      <td>1.330000e+06</td>
      <td>13.000000</td>
      <td>3148.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>651.000000</td>
      <td>174.000000</td>
      <td>1999.000000</td>
      <td>-37.756400</td>
      <td>145.058305</td>
      <td>10331.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>10.000000</td>
      <td>9.000000e+06</td>
      <td>48.100000</td>
      <td>3977.000000</td>
      <td>20.000000</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>433014.000000</td>
      <td>44515.000000</td>
      <td>2018.000000</td>
      <td>-37.408530</td>
      <td>145.526350</td>
      <td>21650.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Data Cleaning

Untuk mengatasi missing value gunakan method `dropna()`,  namun hal ini mempunyai kekurangan yaitu data akan semakin mengecil


```python
df = df.dropna()
df.shape
```




    (6196, 21)



# Tahapan Machine Learning

## Memilih Prediction Target

Disini kita akan mengambil prediction target yaitu `price`


```python
y = df['Price']
y
```




    1        1035000.0
    2        1465000.0
    4        1600000.0
    6        1876000.0
    7        1636000.0
               ...    
    12205     601000.0
    12206    1050000.0
    12207     385000.0
    12209     560000.0
    12212    2450000.0
    Name: Price, Length: 6196, dtype: float64



## Memilih Features

Features digunakan untuk sebagai acuan mesin dalam melakukan pembelajarannya. Dalam memilih features, tidak semuanya kita gunakan hanya beberapa features saja yang digunakan.

Dan features yang akan kita pilih yaitu `Rooms` `Bathroom` `Landsize` `Lattitude` `Longtitude`


```python
features = ['BuildingArea', 'Propertycount', 'Landsize', 'Distance', 'YearBuilt']
X = df[features]
```


```python
X.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BuildingArea</th>
      <th>Propertycount</th>
      <th>Landsize</th>
      <th>Distance</th>
      <th>YearBuilt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>6196.000000</td>
      <td>6196.000000</td>
      <td>6196.000000</td>
      <td>6196.000000</td>
      <td>6196.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>141.568645</td>
      <td>7435.489509</td>
      <td>471.006940</td>
      <td>9.751097</td>
      <td>1964.081988</td>
    </tr>
    <tr>
      <td>std</td>
      <td>90.834824</td>
      <td>4337.698917</td>
      <td>897.449881</td>
      <td>5.612065</td>
      <td>38.105673</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
      <td>389.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1196.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>91.000000</td>
      <td>4383.750000</td>
      <td>152.000000</td>
      <td>5.900000</td>
      <td>1940.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>124.000000</td>
      <td>6567.000000</td>
      <td>373.000000</td>
      <td>9.000000</td>
      <td>1970.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>170.000000</td>
      <td>10175.000000</td>
      <td>628.000000</td>
      <td>12.400000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>3112.000000</td>
      <td>21650.000000</td>
      <td>37000.000000</td>
      <td>47.400000</td>
      <td>2018.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Membuat Model

Disini kita akan menggunakan model `DecisionTreeRegressor`


```python
from sklearn.tree import DecisionTreeRegressor
```

## Konfigurasi Model


```python
df_model = DecisionTreeRegressor(random_state=1)
```

## Melakukan Training Data


```python
df_model.fit(X, y)
```




    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          presort=False, random_state=1, splitter='best')



## Membuat Prediksi


```python
df_model.predict(X.head())
```




    array([1035000., 1465000., 1600000., 1876000., 1636000.])




```python
y.head()
```




    1    1035000.0
    2    1465000.0
    4    1600000.0
    6    1876000.0
    7    1636000.0
    Name: Price, dtype: float64



# Evaluasi Model

## `mean_absolute_error`

Semakin kecil nilai yang dihasilkan oleh MAE, maka hasil prediksi akan menjadi lebih berkualitas


```python
from sklearn.metrics import mean_absolute_error
```


```python
y_hat = df_model.predict(X)
mean_absolute_error(y, y_hat)
```




    553.5829567462879



## Training dan Testing Data


```python
from sklearn.model_selection import train_test_split
```

### Membagi Data Menjadi Dua Bagian


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

### Konfigurasi dan Training Data


```python
df_model = DecisionTreeRegressor(random_state=1)
df_model.fit(X_train, y_train)
```




    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          presort=False, random_state=1, splitter='best')



### Evaluasi


```python
y_hat = df_model.predict(X_test)
mean_absolute_error(y_test, y_hat)
```




    305106.2569399613



## Optimasi Model


```python
def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_hat)
    return mae
```


```python
for max_leaf_nodes in [5, 50, 500, 5000]:
    leaf_mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    print(f'Max Leaf Nodes : {max_leaf_nodes} \t Mean Absolute Error : {int(leaf_mae)}')
```

    Max Leaf Nodes : 5 	 Mean Absolute Error : 326323
    Max Leaf Nodes : 50 	 Mean Absolute Error : 273920
    Max Leaf Nodes : 500 	 Mean Absolute Error : 279725
    Max Leaf Nodes : 5000 	 Mean Absolute Error : 303698
    

# Eksplorasi dengan Model Random Forest


```python
from sklearn.ensemble import RandomForestRegressor
```


```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
rf_model.fit(X_train, y_train)
y_hat = rf_model.predict(X_test)
print(f'Mean Absolute Error : {int(mean_absolute_error(y_test, y_hat))}')
```

    Mean Absolute Error : 218065
    

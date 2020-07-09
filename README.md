# Linear Regression on the World Population

**Completed by Mangaliso Makhoba.**

**Overview:** Use Linear Regression Algorithm to predict a what a country's population may be at a given year. 

**Problem Statement:** Build a model that will make predictions for a specific country at a given year and evalueate the Mean Squared Error.

**Data:** [World Population Dataset (1960 - 2017)](https://raw.githubusercontent.com/Explore-AI/Public-Data/master/AnalyseProject/world_population.csv)

**Deliverables:** A predictive Model

## Topics Covered

1. Machine Learning
3. Linear Regression
5. Mean Squared Error

## Tools Used
1. Python
1. Pandas
2. Scikit-learn
2. Jupyter Notebook

## Installation and Usage

Ensure that the following packages have been installed and imported.

```bash
pip install numpy
pip install pandas
pip install sklearn
```

#### Jupyter Notebook - to run ipython notebook (.ipynb) project file
Follow instruction on https://docs.anaconda.com/anaconda/install/ to install Anaconda with Jupyter. 

Alternatively:
VS Code can render Jupyter Notebooks

## Notebook Structure
The structure of this notebook is as follows:

 - First, we'll load our data to get a view of the predictor and response variables we will be modeling. 
 - Then get select the country we'll be evaluating, with years and population
 - We then split the data into test and train
 - Fit the model and make prediction for any year.
 - Test the model with Mean Squared Error to find out how much the predicted values deviate from the true values.



# Function 1: Country Selection
The world population data spans from 1960 to 2017. We'd like to build a predictive model that can give us the best guess at what the future or past population of a particular country was or might be.

First, however, we need to formulate our data such that sklearn's `Ridge` regression class can train on our data. To do this, we will write a function that takes as input a country name and return a 2-d numpy array that contains the year and the measured population. 

_**Function Specifications:**_
* Should take a `str` as input and return a numpy `array` type as output.
* The array should only have two columns containing the year and the population, in other words, it should have a shape `(?, 2)` where `?` is the length of the data.
* The values within the array should be of type `int`. 

_**Expected Outputs:**_
```python
get_year_pop('Aruba')
```
 ```python
array([[  1960,  54211],
       [  1961,  55438],
       [  1962,  56225],
        ...
       [  2016, 104822],
       [  2017, 105264]])
```

```python
get_year_pop('Aruba').shape == (58, 2)
```



# Function 2: Splitting Data

Now that we have have our data, we need to split this into a training set, and a testing set. But before we split our data into training and testing, we also need to split our data into the predictive features (denoted `X`) and the response (denoted `y`). 

Write a function that will take as input a 2-d numpy array and return four variables in the form of `(X_train, y_train), (X_test, y_test)`, where `(X_train, y_train)` are the features + response of the training set, and `(X-test, y_test)` are the features + response of the testing set.

_**Function Specifications:**_
* Should take a 2-d numpy `array` as input.
* Should split the array such that X is the year, and y is the corresponding population.
* Should return two `tuples` of the form `(X_train, y_train), (X_test, y_test)`.
* Should use sklearn's train_test_split function with a `test_size = 0.2` and `random_state = 42`.

_**Expected Outputs:**_
```python
data = get_year_pop('Aruba')
feature_response_split(data)
```

```python
X_train == array([1996, 1991, 1968, 1977, 1966, 1964, 2001, 1979, 1990, 2009, 2010,
       2014, 1975, 1969, 1987, 1986, 1976, 1984, 1993, 2015, 2000, 1971,
       1992, 2016, 2003, 1989, 2013, 1961, 1981, 1962, 2005, 1999, 1995,
       1983, 2007, 1970, 1982, 1978, 2017, 1980, 1967, 2002, 1974, 1988,
       2011, 1998])

y_train == array([ 83200,  64622,  58386,  60366,  57715,  57032,  92898,  59980,
        62149, 101453, 101669, 103795,  60657,  58726,  61833,  62644,
        60586,  62836,  72504, 104341,  90853,  59440,  68235, 104822,
        97017,  61032, 103187,  55438,  60567,  56225, 100031,  89005,
        80324,  62201, 101220,  59063,  61345,  60103, 105264,  60096,
        58055,  94992,  60528,  61079, 102053,  87277])
        
X_test == array([1960, 1965, 1994, 1973, 2004, 2012, 1997, 1985, 2006, 1972, 2008,
       1963])
       
y_test == array([ 54211,  57360,  76700,  60243,  98737, 102577,  85451,  63026,
       100832,  59840, 101353,  56695])
 ```

# Function 3: Fitting and Prediction

Now that we have formatted our data, we can fit a model using sklearn's `Ridge()` class. We'll write a function that will take as input the features and response variables that we created in the last question, and returns a trained model.

_**Function Specifications:**_
* Should take two numpy `arrays` as input in the form `(X_train, y_train)`.
* Should return an sklearn `Ridge` model.
* The returned model should be fitted to the data.


_**Expected Outputs:**_
```python
train_model(X_train, y_train).predict([[2017]]) == array([[104468.15547163]])
```
# Function 4: Mean Square Error

We would now like to test our model using the testing data that we produced from Question 2. To chieve this, we'll use the mean square error, which for convenience is written as:

<img src="https://render.githubusercontent.com/render/math?math=MSE = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2">



where *p_i* refers to the ith prediction made from `X_test`, *y_i* refers to the ith value in `y_test`, and *N* is the length of `y_test`.

_**Function Specifications:**_
* Should take a trained model and two `arrays` as input. This will be the `X_test` and `y_test` variables from Question 2. 
* Should return the mean square error over the input from the predicted values of `X_test` as compared to values of `y_test`.
* The output should be a `float` rounded to 2 decimal places.

_**Expected Outputs:**_
```python
test_model(lm, X_test, y_test) == 42483684.58
```


## Conclusion
To reduce complexity in the model, the Ridge Regression Algorithm works well in preventing overfitting that simple Linear Rigression is prone to owing to the fact that Linear Models are Bias in general because they are least flexible with the least degrees of freedom. The Ridge regressor counters data complexity by reducing the weights of features that may not be contributing in the prediction. 

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

## Contributing Authors
**Authors:** Mangaliso Makhoba, Explore Data Science Academy

**Contact:** makhoba808@gmail.com

## Project Continuity
This is project is ongoing


## License
[MIT](https://choosealicense.com/licenses/mit/)
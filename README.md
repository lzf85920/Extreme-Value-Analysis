# Extreme Value Analysis 

## What is outlier ?
In statistics, an outlier is a data point that differs significantly from other observations. An outlier may be due to variability in the measurement or it may indicate experimental error; the latter are sometimes excluded from the data set. An outlier can cause serious problems in statistical analyses.

### Outlier can be of two types: 
1. **Univariate** : Univariate outliers are extreme values in the distribution of a specific variable
2. **Multivariate** : Multivariate outliers are a combination of values in an observation (n-dimensional space)

## What reason cause for outliers ?
* **Data Entry Errors** : Human errors such as errors caused during data collection, recording, or entry can cause outliers in data.
* **Measurement Error** :  It is the most common source of outliers. This is caused when the measurement instrument used turns out to be faulty.
* **Natural Outlier** : When an outlier is not artificial (due to error), it is a natural outlier. Most of real world data belong to this category.

## Detection outlier 
1. Hypothesis Testing (Grubbs's test)
2. Z-score method
3. **Robust Z-score**
4. **I.Q.R method**
5. Winsorization method(Percentile Capping)
6. DBSCAN Clustering
7. **Isolation Forest**
8. **Visualizing the data**


---

**Robust Z-score (Modified Z-score method)**

It is also called as Median absolute deviation method. It is similar to Z-score method with some changes in parameters. Since **mean and standard deviations are heavily influenced** by outliers, alter to this parameters we use median and absolute deviation from median.

![](https://i.imgur.com/nRkThqD.png)


We can calculate the Robust Z-score like this:

![](https://i.imgur.com/xr2svfH.png)

```
# Robust Z-score
def Robust_zscore(df):
    med = df.median()
    mad = stats.median_abs_deviation(df)
    zscore = (0.6745*(df-med))/mad
    outler = zscore[zscore > 3]
    return outler
```
---

**Tukey’s box plot method (I.Q.R method)**

IQR = Q3 - Q1
![](https://i.imgur.com/HSmfc0O.png)

```
# IQR method
def IQR_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outler = df[(df > upper)|(df < lower)]
    return outler
```
---

**Isolation Forest**

Isolation forest is a machine learning algorithm for anomaly detection (based on Decision tree). It's an unsupervised learning algorithm that identifies anomaly by isolating outliers in the data. **Which not only detects anomalies faster but also requires less memory compared to other anomaly detection algorithms.**
![](https://i.imgur.com/gtEQCNs.png)

 As anomalies data points mostly have a lot shorter tree paths than the normal data points, trees in the isolation forest does not need to have a large depth so a smaller max_depth can be used resulting in low memory requirement.

```
# Isolation Forest
model = IsolationForest(max_features = 1
, n_estimators = 50, contamination = 'auto', random_state = 1)
model.fit(df[['DIS']])
df['score']=model.decision_function(df[['DIS']])
df['anomaly']=model.predict(df[['DIS']])
anomaly = df[['DIS', 'score', 'anomaly']]
anomaly[anomaly['anomaly'] == -1]
```


---

**Visualizing the data**
1. box plot
2. Scatter plot
3. Histogram
4. Distribution Plot
5. QQ plot
```
# box plot
df_2 = df[['CRIM', 'ZN', 'INDUS', 'RM', 'AGE', 'DIS', 'RAD', 'PTRATIO','LSTAT']]
ax = sns.boxplot(data=df_2, orient="h", palette="Set2")
```
![](https://i.imgur.com/K6FoziH.png)
```
# Scatter plot
plt.scatter(df['DIS'],df['ZN'])
```
![](https://i.imgur.com/vYC1gQX.png)
```
#Hist Plot
plt.hist(df['ZN'])
```
![](https://i.imgur.com/LgpSc49.png)
```
# Dist Plot
sns.distplot(df['DIS'], bins=20)
```
![](https://i.imgur.com/uwBbapK.png)
```
# QQ Plot
sm.qqplot(df['DIS'], line = '45', fit = True) 
```
![](https://i.imgur.com/5TMm3ei.png)


## Handling Outliers 
1. Deleting observations
2. **Transforming values**
3. **Imputation**
4. Separately treating


---

**1. Transforming values**
* Scalling
* Log transformation
* Cube Root Normalization
* Box-Cox transformation
* Yeo-Johnson Transformation



---

**(1) Box-Cox transformation**

A Box Cox transformation is a transformation of a non-normal dependent variables into a normal shape. Normality is an important assumption for many statistical techniques; if your data isn’t normal, applying a Box-Cox means that you are able to run a broader number of tests.
![](https://i.imgur.com/ZWhDUOu.png)
```
# Box-Cox Transformation
bcx_DIS, lam = boxcox(df['DIS'])

plt.figure(figsize=(10, 3))

plt.subplot(121)
sns.distplot(df['DIS'], bins=20)
plt.title('Orginal')

plt.subplot(122)
sns.distplot(bcx_DIS, bins=20)
plt.title('Box-Cox transformation')
plt.xlabel('DIS trans')
```
![](https://i.imgur.com/uj5JobO.png)



---


**(2) Yeo-Johnson Transformation**

The Yeo–Johnson transformation **allows also for zero and negative values of $y$**. $\lambda$  can be any real number, where $\lambda$ =1 produces the identity transformation. The transformation law reads:
![](https://i.imgur.com/PzlbOzA.png)

```
# Yeo-Johnson Transformation
bcx_ZN, lam = yeojohnson(df['ZN'])

plt.figure(figsize=(10, 3))

plt.subplot(121)
sns.distplot(df['ZN'], bins=20)
plt.title('Orginal')

plt.subplot(122)
sns.distplot(bcx_ZN, bins=20)
plt.title('Yeo-Johnson Transformation')
plt.xlabel('ZN trans')
```
![](https://i.imgur.com/WKcn51L.png)


---

**2. Imputation**

Like imputation of missing values. We can use mean, median or zero value to replace outlier.


---

## Handling Missing Values

**1. Counting Missing Values**
```
missing_values_count = df.isnull().sum()
```
**2. Imputation (Numerical Missing Data)**

```
# Mean
data = np.array([[1, 2], [np.nan, 3], [7, 6]])
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data)
```
![](https://i.imgur.com/dOCU47A.png)

```
# Median
data = np.array([[1, 2], [np.nan, 3], [7, 6],[2,8]])
imp = SimpleImputer(missing_values=np.nan, strategy='median')
data = imp.fit_transform(data)
```
 ![](https://i.imgur.com/6ivJfIv.png)

```
# Most Frequent
data = np.array([[1, 2], [np.nan, 3], [7, 6],[7,8]])
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data = imp.fit_transform(data)
```
![](https://i.imgur.com/h0YLovg.png)

```
# Constant
data = np.array([[1, 2], [np.nan, 3], [7, 6],[7,8]])
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=5)
data = imp.fit_transform(data)
```
![](https://i.imgur.com/Y1wHO8Z.png)

**2. Imputation (Categorical Missing Data)**
For handling categorical missing values, you could use one of the following strategies. However, it is the "most_frequent" strategy which is preferably used.
* Most frequent (strategy='most_frequent')
* Constant (strategy='constant', fill_value='someValue')






---

## References
1. https://medium.com/james-blogs/outliers-make-us-go-mad-univariate-outlier-detection-b3a72f1ea8c7
2. https://www.kaggle.com/nareshbhat/outlier-the-silent-killer/comments
3. https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755
4. https://www.statisticshowto.com/upper-and-lower-fences/
5. https://blog.paperspace.com/anomaly-detection-isolation-forest/
6. https://heartbeat.fritz.ai/isolation-forest-algorithm-for-anomaly-detection-2a4abd347a5
7. https://pubs.rsc.org/tr/content/articlelanding/2016/ay/c6ay01574c#!divAbstract
8. https://dzone.com/articles/imputing-missing-data-using-sklearn-simpleimputer
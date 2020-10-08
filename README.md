# Master DSI notes


 ## Relevant links 
* [Course Page](https://github.com/GalvanizeDataScience/course-outline/tree/20-10-DS-DEN_DEN19)

* [Quick Reference guide](https://github.com/GalvanizeDataScience/course-outline/tree/20-10-DS-DEN_DEN19/quick-reference)

* [Pandas Groupby syntax](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html)



--------------------------------------------
## Del, Topics to study each night
* Pandas Data frames



-----------------------

## Markdown style Guide
* One '#' is used for the biggest size text
* Two '##' is used for second biggest size text
* To post a link use ['link title'](linkgoes here)
* To use bullet point use '*'
* To make a code block use ' ``` '
* Images - !['title of image']('link to image')


-----------------------

# Python

## Data wrangling with pandas workflow
1. Build a DataFrame from the data (ideally put all the data in this object)

```python
names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv('imdb_top_10000.txt', delimiter='\t', names=names).dropna()
print "Number of rows: %i" % data.shape[0]
data.head()  # print the first 5 rows
```
2. Clean the DataFrame. It should have the following properties:
* Each row describes a single object
* Each column describes a property of that object
* Columns are numeric whenever appropriate
* Columns contain atomic properties that cannot be further decomposed
3. Explore global properties. Use histograms, scatter plots, and aggregation functions to summarize the data.
4. Use groupby and small multiples to compare subsets of the data. 
```python
#mean score for all movies in each decade
decade_mean = data.groupby(decade).score.mean()
decade_mean.name = 'Decade Mean'
print decade_mean

plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)
remove_border()
```



## Libraries

### Pandas
* Indexing

```python
import pandas as pd # Standard import
```

```python
# Create a dataframe using pandas

import pandas as pd # I haven't actually done this in code yet. 
data_lst = [{'a': 1, 'b': 2, 'c':3}, {'a': 4, 'b':5, 'c':6, 'd':7}]
df = pd.DataFrame(data_lst)
df

Out[]:	a	b	c	d
0	1	2	3	NaN
1	4	5	6	7.0
```

```python
# Reading in data with pandas/ reading external data/ import data
df = pd.read_csv('my_data.csv')

# if no header name
df = pd.read_csv('my_data.csv', header=None)

df = pd.read_csv('my_data.csv', header=None, names=['col1', 'col2', ...., 'col12'])
```

```python
# How to clean data / cleaning data with pandas

# Clean column names
df.columns

cols = df.columns.tolist()
cols = [col.replace('#', 'num') for col in cols] # this replaces the pound sign with num
cols = [col.replace(' ', '_'.lower())]
print(cols)

```
```python
# masking in pandas 
df['chlorides'] <= 0.08 # This just gives us a mask - tells us True or False whether each row 
                 # fit's the condition.

# To use a mask, we actually have to use it to index into the DataFrame (using square brackets). 
df[df['chlorides'] <= 0.08]


# Okay, this is cool. What if I wanted a slightly more complicated query...
df[(df['chlorides'] >= 0.04) & (df['chlorides'] < 0.08)]

```
```python
# creating and dropping columns in pandas
# how to create a column in pandas
# how to delete a column in pandas / delete column / drop column
df['non_free_sulfer'] = df['total sulfur dioxide'] - df['free sulfur dioxide'] # add a new column titled non_free sulfer

df.drop('non_free_sulfur2', axis =1. inplace = True) # Drop the non_free_sulfur2 column. Axis = 1 is referring to axis 1 which is columns

```
```python
# Null values in pandas / how to get rid of missing values / how to fill missing values in pandas
df.fillna(-1, inplace=True)
df.dropna(inplace=True) # Notice the addition of the inplace argument here. 
```


```python
# Cast the date column as datetime object / how to add date and time with pandas

df['incident_date'] = pd.to_datetime(df['incident_date'])

```

```python
# groupby in pandas / how to use grouby in pandas

df.groupby('quality') # Note that this returns back to us a groupby object. It doesn't actually 
                      # return to us anything useful until we perform some aggregation on it. 

groupby_obj = df.groupby('quality')
groupby_obj.mean()
groupby_obj.max()
groupby_obj.count()
# pull a specfic column from quality
df.groupby('quality').count()['fixed_acidity'] 

# We can also apply different groupby statistics to different columns...
# here we are grouping by quality but using a dictonary to return different aggregate functions upon different columns. Useful!
df.groupby('quality').agg({'chlorides': 'sum', 'fixed_acidity': 'mean'})
```
```python
# sorting in pandas / how to sort in pandas

df.sort_values('quality') # Note: this is ascending by default.
df.sort_values('quality', ascending=False)


df.sort_values(['quality', 'alcohol'], ascending=[True, False]) # ascending=False will apply to both columns. 
```
```python
# how to reset index in pandas / reset index in pandas
df.sort_values(['quality', 'alcohol'], ascending=[True, False].reset_index())
# reset index is extremely useful when using groupby function
```
```python
# how to set an index in pandas
df_new.set_index(['three', 'four', 'five'])
```

```python
# Useful dataframe attributes

df.shape # gives the number of rows and cols

df.columns # gives back a list of all column names

df.describe()  # gives summary statistics for all numeric cols

df.head() # shows you the first n rows (n=5)

df.tail() # shows the last n rows (n-5)

```

```python
# How to rename a column in pandas
# Rename an individual column in the original data frame. 

df.rename(columns={'fixed acidity': 'fixed_acidity'}, inplace=True)
print(df.columns)

```
```python
# replace all spaces with underscored in dataframe using pandas
df2 = df.copy()
cols = df2.columns.tolist()
cols = [col.replace(' ', '_') for col in cols]
df2.columns = cols
df2.volatile_acidity
```
```python
# acess a certain column in a dataframe using pandas
df['chlorides'] # Grabs the 'chlorides' column. 
```
```python
# accessing multiple columns by passingin a list of column names pandas

df[['chlorides', 'volatile acidity']]
```
```python
# row indexing using pandas

df[:3] # This will grab from the beginning up to but not including the row at index 3. 

# This will grab up to but not including the row at index 1 (i.e. it'll grab the row  at index 0). 
df[:1]

```
```python
# using loc and iloc in pandas
# .loc is looking for lables or location
# .iloc is looking for indicies
# .iloc is non-inclusive
# .loc is inclusive

df.loc[0, 'fixed_acidity'] # 0 is one of the index labels, and 'fixed acidity' is a column label.

# Ranges on our index labels still work (as long as they're numeric).
df.loc[0:10, 'fixed_acidity']


df.loc[10:15, ['chlorides', 'fixed_acidity']]
```
```python
# how to combine datasets with pandas / combining datasets in pandas
# get_dummies is a method called on the pandas module - you simply pass in a Pandas Series 
# or DataFrame, and it will convert a categorical variable into dummy/indicator variables. 
quality_dummies = pd.get_dummies(wine_df.quality, prefix='quality')
quality_dummies.head()

#Now let's look at the join() method. Remeber, this joins on indices by default. This means that we can simply join our quality dummies dataframe back to our original wine dataframe with the following...

joined_df = wine_df.join(quality_dummies)
joined_df.head() 

# Let's now look at concat. concatanate two dataframes together
joined_df2 = pd.concat([quality_dummies, wine_df], axis=1)
joined_df2.head()

# Using pd.merge
pd.merge(red_wines_quality_df, white_wines_quality_df, on=['quality'], suffixes=[' red', ' white'])
```

```python
# Selecting a subset of columns

In [9]: df[['float_col','int_col']]
Out[9]:
   float_col  int_col
0        0.1        1
1        0.2        2
2        0.2        6
3       10.1        8
4        NaN       -1


# Conditional indexing

In [7]: df[df['float_col'] > 0.15]
Out[7]:
   float_col  int_col str_col
1        0.2        2       b
2        0.2        6    None
3       10.1        8       c
```
* Renaming Columns
    
```python
In [9]: df2 = df.rename(columns={'int_col' : 'some_other_name'})

Out[10]:
   float_col  some_other_name str_col
0        0.1                1       a
1        0.2                2       b
2        0.2                6    None
3       10.1                8       c
4        NaN               -1       a



```
* Handling missing values
```python
n [12]: df2
Out[12]:
   float_col  int_col str_col
0        0.1        1       a
1        0.2        2       b
2        0.2        6    None
3       10.1        8       c
4        NaN       -1       a

In [13]: df2.dropna()
Out[13]:
   float_col  int_col str_col
0        0.1        1       a
1        0.2        2       b
3       10.1        8       c
```
* Fill missing values
```python
In [16]: df3
Out[16]:
   float_col  int_col str_col
0        0.1        1       a
1        0.2        2       b
2        0.2        6    None
3       10.1        8       c
4        NaN       -1       a

In [17]: df3['float_col'].fillna(mean)
Out[17]:
0     0.10
1     0.20
2     0.20
3    10.10
4     2.65
Name: float_col
```
* Groupby
```python
In [41]: grouped = df['float_col'].groupby(df['str_col'])

In [42]: grouped.mean()
Out[42]:
str_col
a           0.1
b           0.2
c          10.1
```

```python
# how to graph in pandas
# NOTE- Do not give presentations using pandas plots. use matplotlib for this.

df.plot(kind = 'hist')
df.hist(figsize = 10) # shows a lot of histograms.  
df['quality'].plot(kind = 'hist')
df.plot(kind = 'scatter', x = 'x_var', y= 'y_var')
df.plot(kind ='box')
```

-----------------------------------------
## Matplotlib

```python
import matplotlib.pyplot as plt
```
* Basic line
```python
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()
```
* Scatter plot
```python
plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()
```
* Plotting with categorical variables
```python
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132) # note the sublotting syntax here
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()
```
* Sub plot syntax
```python
plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])


plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3') # subplot 211 title
```
* Annotating text (how to place text)
```python

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             ) #placement occurs here

plt.ylim(-2, 2)
plt.show()
```







------------------------------


# Math 
## Linear Algebra Introduction Lecture
* Scalar: A quantity that only has a magnitude (NO DIRECTION)
* Vector: A quantity that has a magnitude and a direction. Any array is an ordered sequence of values. A vector is an array.
Axes (columns) of data as directions that define the vector space of data.
* Matrices: A matrix is a rectangular array of numbers arranged in rows and columns. 
Shape of a matrix is the number of rows and number of columns
* Euclidean Norm: (Euclidean distance) or L2 norm. 
* Dot product: An operation that takes two equal length vectors and returns a single number. 
* Matrix multiplication can only happen when the number of a column in the firsst matrix matches the number of rows in the second matrix.




# Master DSI notes


 ## Relevant links 
* [Course Page](https://github.com/GalvanizeDataScience/course-outline/tree/20-10-DS-DEN_DEN19)

* [Quick Reference guide](https://github.com/GalvanizeDataScience/course-outline/tree/20-10-DS-DEN_DEN19/quick-reference)

* [Pandas Groupby syntax](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html)

* [Matplotlib intro syntax]()

* [Pandas lecture](http://localhost:8888/notebooks/post-lecture_intro_pandas_notes.ipynb)

* [Pandas syntax page](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)


-----------------------------------------------------

## Career services

* [Deliverables Dashboard](https://docs.google.com/spreadsheets/d/12de6aB2Yx0S47aQZsFyVX4xf1hBwftkaWRikw19mRWQ/edit#gid=1543183776)


2 - intro to career serviecs
3 - linkedin
5 - resumes
6 - inortmational interviews
7 - professional introductions & building rapport
8 - cover letters and interview prep
9 - public speaking and job search strategy
10 - behavioral mock interviews and Salary negotiation

## Normal Imports

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')
```


--------------------------------------------
## Del, Topics to study each night
* Pandas Data frames
* Stats Quest for breakdown of stats concepts


-----------------------

## Markdown style Guide
* One '#' is used for the biggest size text
* Two '##' is used for second biggest size text
* To post a link use ['link title'](linkgoes here)
* To use bullet point use '*'
* To make a code block use ' ``` '
* Images - !['title of image']('link to image')

-----------------------------------------
## Week 2 Student review

1. What distribution would you use in the following cases:

What is the probability that the mean volume of 50 bottles is less than 500 ml? Normal distribution

Deciding to go for a run or not. Bernouli

Determining how many days pass before you finally decide to go for a run. Hypergeometric Distribution

Determining how likely it is that you go for 10 runs in a month. Binomial

Calculating which day of the month you buy new shoes. Possibly uniform, as it could be completely random giving the lack of information provided. Could be geometric as well.

Assuming you run at a 9 minute mile avg pace, determining how likely it is that you pass the 3 mile mark in a race in 25 minutes?

Poisson.





2. What is the central limit theorem?
The central limit theorem allows us to make probabilistic statements about the sample mean from any population using the normal distribution.

3. A bootstrap sample from a sataset is a sample taken with REPLACEMENT from the data set whose size is the size of the dataset itself. 

4. Here are the numbers in a list:
[3, 4, 5, 7, 8, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22, 23, 24, 25, 26, 31, 42, 60, 69, 86, 108, 256]

For this list:
a) Find a 90% confidence interval for the mean using the Central Limit Theorem.

def confidence_interval(mean, st_err, num_std):
    ci = [mean - num_std * st_err, mean + num_std * st_err]
    return [round(val, 3) for val in ci

b) Find a 90% confidence interval for the mean using bootstrapping.
c) Find a 68% confidence interval for the standard deviation using the Central Limit Theorem.
d) Find a 68% confidence interval for the standard deviation using bootstrapping.
e) Either a), b), c) or d) is a trick question. Which one is it, and why?


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
# how to create columns and rows with new data
# how to make columns and rows with new data
# Creating columns are rows with new data

data_vals = [[1, 2, 3], [4, 5, 6]]
data_cols = ['a', 'b', 'c']
df = pd.DataFrame(data=data_vals, columns=data_cols)
df

```

## What to do when you first get data with pandas


```python



# first steps to take when working with data in pandas
# how to begin with pandas
# step by step guide on how to start with pandas

# 1: Start with the basics - exploring the data

    df = pd.read_csv('data/winequality-red.csv')
    df.head()
    df.shape  # give you the number of rows and number of cols
    df.columns # gives you back a list of all the column names
    df.info()  # allows you to look at the data type for each column and the number of null values.
    df.describe() # gives you summary stats for all of your numeric columns
    df.tail() # shows you the lastr n rows
    df.unique()
    df['column name'].unique() # gives unique values in that column


# 2: Clean your data!

    df2 = df.copy() # best practice to make a copy of the original data
    cols = df2.columns.tolist()   # All this does is takes the column and put its content into a list format
    cols = [col.replace(' ', '_') for col in cols] # takes the list format and replaces all empty spaces with an underscore. ---- how to replace spaces with underscore in pandas
    df2.columns = cols  # THIS IS IMPORTANT. You need to re-assign the manipulated column back into the data frame.


    # If you need to rename a column in the original dataframe---
    # how to rename a column with pandas
    df.rename(columns={'original column name' : 'the_column_name_i_want_it_to_be_named_to'}, inplace = True)     # DONT FORGET THE INPLACE = TRUE!


# 2.5: Dealing with NULL data.
# how to get rid of null values
# how to fix missing values

    df.fillna(-1, inplace= True)  # this will fill all NA/NULL values with the value -1
    df.dropna(inplace = True)    # this will just drop all column/spaces that have a NULL value


     
    


# 3: Acessing your data!
    df['column name'] # grabs the column titled 'column name'

    df[['column1' , 'column2']]   # how to access multiple columns 

    df[:3] # this will grab from the beginning up to but not including the row at index 3

    df[:1] # this will grab up to but not including 



# 4: How to mask your data
df['column_name'] <= 0.08 # this just gives us a mask, tells us True or false wether each row fits the condition

df[df['column_name'] <= 0.08 ] # proper way to use a mask 
df[(df['column_name'] >= 0.04) & (df['column_name'] < 0.08)] # a more complicated mask

# 5: How to create new columns
df['new column'] = df['old column'] - df['old column2']


# 6: Groupby/groupby/ how to groupby in pandas

# when grouping you need to assign the group to a value or else it wil return the memory location.
# when you group, the column you group by becomes the index. It will return all columns based on the groupby'd column

    groupby_obj = df.groupby('column_name')
    groupby_obj.mean()
    groupby_obj.max()
    groupby_obj.count()

    # if you want to get a group by and a specific column:

    df.groupby('column_name').count()['column_name2]

    # how to group by multiple columns
    
    df.groupby(['column1', 'column2']).count()['column3']

    # how to apply statistics to a group by
    # how to group by an equation
    # advanced groupby

    df_new = df.groupby('column1').agg({'column2': 'sum', 'column3' : 'mean'}).reset_index()


# 7: How to sort using pandas
    # sorting data in pandas

    df.sort_values('column_name') # default here is ascending
    df.sort_values('column_name', ascending = False)


    # a more specific sort

    df['column_name'].sort_values(ascending = False) 

    # you can sort by multiple columns by placing them in a list
    # how to sort by multiple columns in pandas
    # it will sort the first column passed first, then the second

    df.sort_values(['column_1', 'column2'], ascending =[True, False]).reset_index(drop = True)



#8 how to combine datasets with pandas
    # how to join data sets in pandas


# first you need to look at the dummies method. 

column_name_dummies = pd.get_dummies(dataframe_df.columnname, prefix='quality')

# now to actually join. How to join datasets

    joined_df = dataframe_df.columnname.join(column_name_dummies)

# how to concat two dataframes together
# how to concatonate with pandas

joined_df2 = pd.concat([column_name1, df_dataframe], axis = 1)

# how to merge with pandas

pd.merge(red_wines_quality_df, white_wines_quality_df, on=['quality'], suffixes=[' red', ' white'], how = 'outer')
```







```python
# Reading in data with pandas/ reading external data/ import data
df = pd.read_csv('my_data.csv')
data = np.loadtxt('my_data.csv)


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
```python
# standard default style you should use for matplot
plt.style.use('ggplot') # I also like fivethirtyeight'
matplotlib.rcParams.update({'font.size': 16, 'font.family': 'sans'})
```
```python

# plt example
x_data = np.arange(0, 4, .011)

y1_data = np.sin(x_data)
y2_data = np.cos(x_data)

plt.subplot(2, 1, 1)      # #rows, #cols, plotnumber
plt.plot(x_data, y1_data) # First plot
plt.plot(x_data, y2_data) # Second plot in same subplot
plt.title('using plt')    # plot title

plt.subplot(2, 1, 2)                 # #rows, #cols, plotnumber
plt.plot(x_data, y2_data, color='r') # Second plot in a separate subplot
plt.xlabel('x')

plt.show()
```
```python
# how to iterate through axes in matplot
# super important matplotlib function!
fig, axs = plt.subplots(2, 3)

for i, ax in enumerate(axs.flatten()):
    string_to_write = f"Plot: {i}"
    ax.text(0.1, 0.4, string_to_write, fontsize=20)
    #       ^ These were figured out through trial and error.
    
fig.tight_layout()

```
```python
# how to write some text to a single axis
fig, ax = plt.subplots(figsize=(8, 2))
ax.text(0.35, 0.4, "Hi Y'all!", fontsize=35)
```
```python
# titles and labels in matplotlib
# how to add a label in matplotlib
# how to add a title in matplotlib
fig, ax = plt.subplots()
ax.set_title("This is a Great Plot!")

# each axes can have its own title
# how to give each subplot its own title in matplotlib

fig, axs = plt.subplots(2, 3)

for i, ax in enumerate(axs.flatten()):
    ax.set_title(f"Plot: {i}")
fig.tight_layout()


# axis labels / how to set labels in matplotlib
# Axis labels use ax.set_xlabel and ax.set_ylabel.

fig, ax = plt.subplots()
ax.set_title("This is a Great Plot!")
ax.set_xlabel("This is an x-label")
ax.set_ylabel("This is a y-label!")
```
```python
# scatter plots in matplotlib
# how to make a scatterplot in matplotlib
x = np.random.normal(size=50)
y = np.random.normal(size=50)
fig, ax = plt.subplots()
ax.scatter(x, y)

# when there is a lot of points in a scatter plot uses alpha
fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.3)  # alpha controls transparency (0 is transparent; 1 is opaque)

```
```python
# linear regression data in matplotlib
```
```python
# how to add color to your graph matplotlib
# adding color matplotlib
# Getting data
x_blue = np.random.uniform(size=100)
y_blue = 1.0*x_blue + np.random.normal(scale=0.2, size=100)

x_red = np.random.uniform(size=100)
y_red = 1.0 - 1.0*x_red + np.random.normal(scale=0.2, size=100)

# plotting
fig, ax = plt.subplots()
ax.scatter(x_blue, y_blue, color="blue")
ax.scatter(x_red, y_red, color="red")
```
```python
# drawing line plots 
# line plots in matplotlib
x = [0, 1, 2, 3, 4]
y = [-1, 0, 1, 0, 1]

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.5)
```
```python
# plot linear in matplotlib
# plot quadratic in matplotlib
# plot cubic in matplot lib
linear = lambda x: x - 0.5
quadratic = lambda x: (x - 0.25)*(x - 0.75)
cubic = lambda x: (x - 0.333)*(x - 0.5)*(x - 0.666)
functions = [linear, quadratic, cubic]

# Set up the grid.
x = np.linspace(0, 1, num=250)

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
for f, ax in zip(functions, axs.flatten()):
    ax.plot(x, f(x))
```
```python
# linear regression in matplotlib
slopes = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
x_linspace = np.linspace(-1, 1, num=250)

fig, axs = plt.subplots(2, 3, figsize=(10, 4))

for i, ax in enumerate(axs.flatten()):
    x = np.random.uniform(-1, 1, size=50)
    y = slopes[i]*x + np.random.normal(scale=0.2, size=50)
    ax.plot(x_linspace, slopes[i]*x_linspace, linewidth=2.5)
    ax.scatter(x, y, color="blue")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Slope: {:2.2f}".format(slopes[i]))
    
fig.tight_layout()
```
```python
# Drawing bar charts
# how to make a bar chat in matplotlib

record_counts = pd.DataFrame(
    {'count': [135, 40, 20, 30, 15], 
     'genre': ['rock', 'metal', 'jazz', 'rap', 'classical']}
)
# x will be the left hand edge of the bars.
x = np.arange(len(record_counts['genre']))

fig, ax = plt.subplots()

ax.bar(x, record_counts['count'])
# Make the ticks at the center of the bar using:
#   center = left_edge + 0.5*width
ax.set_xticks(x)
ax.set_xticklabels(record_counts['genre'])
```
```python
# code to show the different plot styles
# plot styles in matplotlib
# how to stlye your graph in matplotlib

x = np.random.rand(100)
y = 4 +3*x*np.random.rand(100)
for style in plt.style.available[:13]:
fig = plt.figure(figsize=(8,8))
plt.style.use(style)
plt.scatter(x,y)
plt.title(f'{style}', fontweight='bold', fontsize= 16)
```
```python
# function to draw a scatter plot
def draw_scatterplot(ax, center, size, color):
    x = np.random.normal(loc=center[0], size=size)
    y = np.random.normal(loc=center[1], size=size)
    ax.scatter(x, y, color=color, label=color+' data points')

# function to draw a line matplotlib
def draw_line(ax, x_range, intercept, slope):
    x = np.linspace(x_range[0], x_range[1], num=250)
    ax.plot(x, slope*x + intercept, linewidth=2, label='decision boundary')

```
```python
# location legend in matplotlib

Location String	Location Code
'best'	0
'upper right'	1
'upper left'	2
'lower left'	3
'lower right'	4
'right'	5
'center left'	6
'center right'	7
'lower center'	8
'upper center'	9
'center'	10

* bbox_to_anchor to controls the location of the box in coordination with loc. You can use a '2-tuple' or '4-tuple' of floats. The '2-tuple' controls the (x,y) coordinates of box. The '4-tuple' controls the (x,y,width,height)


```
```python
# how to set twin axes in matplotlib
fig, ax = plt.subplots()
ax.plot(x,np.exp(x))
ax.set_ylabel('exp')
ax.set_xlabel('x')
ax2 = ax.twinx()
ax2.plot(x, np.log(x),'bo-') # 'bo-': blue circle connected via line
ax2.set_ylabel('log')
ax.set_title('Exponential and Log Plots')
fig.legend(labels = ('exp','log'),loc='upper left');
```
```python
# heatmap example


vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots(figsize=(12,12))
im = ax.imshow(harvest) # this is actually how you show the heat map

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45) #, ha="right") #,  # setp --> set preferences
                            

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j], color="w",fontsize=14)
                       #ha="center", va="center", color="w",fontsize=14)
# va -->'center' | 'top' | 'bottom' | 'baseline'
# ha -->'center' | 'right' | 'left' 
                                                    

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
```
```python
# histogram in matplotlib
# how to create a histogram in matplotlib
N_points = 100000
n_bins = 20

# Generate a normal distribution, center at x=0 
x = np.random.randn(N_points)

fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(x, bins=n_bins)
axs[1].hist(x, bins=n_bins, density=True);
```

 
## Scipy Stats Module

```python

# How to import scipy and make it look pretty
# make scipy loook pretty


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Always make it pretty.
plt.style.use('ggplot')
font = {'weight': 'bold',
        'size':   16}
plt.rc('font', **font)
```


```python
# Scatter plot with jitter
def one_dim_scatterplot(data, ax, jitter=0.2, **options):
    ## why jitter? especially for bootstraping later
    if jitter:
        jitter = np.random.uniform(-jitter, jitter, size=data.shape)
    else:
        jitter = np.repeat(0.0, len(data))
    ax.scatter(data, jitter, **options)
    ax.yaxis.set_ticklabels([])
    ax.set_ylim([-1, 1])
    ax.tick_params(axis='both', which='major', labelsize=15)


fig, ax = plt.subplots(1, figsize=(12, 1))
one_dim_scatterplot(data, ax, s=15)
```



```python
# The Empirical Distribution function
def emperical_distribution(x, data):
    weight = 1.0 / len(data)
    count = np.zeros(shape=len(x))
    for datum in data:
        count = count + np.array(x >= datum)
    return weight * count
```







```python
unioform_dist = stats.randint(low=0, high =10)
benoulli = stats.bernoulli(p=0.4)
binomial = stats.binom(n =50, p=0.4)
hypergeom = stats.gypergeom(M=20, n=7, N=12)  #non standard parameters
poisson = stats.poisson(mu = 5) # mu is the same as lamda

# contonous distributions
uniform_cont = stats.uniform(loc-0, scale= 10)   #non-standard parameters
normal = stats.norm(loc=0.0, scale=1) #non standard params
exponental = stats.expon(loc=2.0)
```

```python
# Calculating CDF / how to calculate cdf

print("P(Binomial(n = 50, p -0.4) <=20) = ", binomial.cdf(20))
print("p(Normal(mu=0.0, sigma= 1.0) <= 1.0 =", normal.cdf(1,0)))


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

## Probability distribution lecture
### Discrete Distributions
* Probability mass fucntion: PMF of a discrete distribution gives the probability of observing each possible outcome(x) of the random variable X
* Cumulative distribution function - is the probability of observing an outcome less than or equal to x

--------------------------

### Continous distributions
* continous distributions have a cdf that is smooth, not jumpy.
* Probability density function (PDF) 
* We can't take a running sum of the PDF to get a cdf, but in calculus there is integration. Think of it as taking the area under the curve.  

* pmF calculates the probability of a sinlge event happening
* the cdf calculates the probability of that one event and everything less than that. 


### Types of Discrete Distributions
#### Uniform Distribution
* Describes a situation with a fininte number of outcomes, where each outcome is as equally likely as any other.  THIS IS ONLY MENT TO BE A SINGLE EVENT.       ex: a die roll

#### Bernouli Distribution
* Simplest distribution. It is a model of a single flip of a coin
* there are only two possible outcomes for X, usually 1 or 0

#### Binomial Distribution
* is a counting distribution. It models flipping a coin multiple times and counting the number of certain outcomes.
* parameters = n,p where n is the number of trials and p is the probability

#### Hypergeometric Distribution
* another counting distribution. The probability that a succes will occuron or before the case trial. 

ex: This one models a deck of cards of two types(red/blue). If you shuffle the deck, draw some number of cards, and then count how many blue cards you have, this count is hyper geometrically distributed.


* parameters = n- total sample size          k = total number of blue cards in deck    n = size of the hand you drew


#### Poisson Distribution
* Counting distribution
* Models a process where events happen at a fixed rate or frequency and you're watching it for a fixed amount of time

* ONL APPLICABLE if the events are independent and identically distributed
* parameters = lambda 

### Continous distributions


#### Uniform Distribution
* A set of outcomes that are all equally likely, but this time any number in an interval is a possible output of the random variable.


#### Normal Distribution (Gauissian)
* primary importance in probability and stats theory due to the Central Limit theorem.
* how to calculate mu for a nomral distribution : (N * P)

#### Exponential Distribution
* contionous distribution related to the Poisson distribution
* How much time will it take to observe the first event.
* ex : Students arrive at a local bar and restaurant at a mean rate of 1 student every 10 minutes. What is the probability that the next student will enter the bar within the next 5 minutes?






* distribution paramaters
```python
# paramaters for different distributions
binomial_samp = make_draws(stats.binom, {'n': 100, 'p':0.1})
poisson_samp = make_draws(stats.poisson, {'mu' : 2})
exponential_samp = make_draws(stats.expon, {'scale' :2})
geometric_samp = make_draws(stats.geom, {'p': 0.1})
```

----------------------------------------------------------

## Binomial tests
* A binomial test is a specific type of hypothesis test involving random variables that can be modeled by the binomial distribution

1. State a scientific question - its answer should be yes or no
2. State a null hypothesis - the skeptic’s answer to the question -- the status qho
3. State an alternative hypothesis - the non-skeptic’s answer to the question
4. Create a model - the model assumes the null hypothesis is true
5. Set a threshold - decide how surprised you need to be to reject the null
hypothesis
6. Collect your data
7. Calculate a p-value - the probability of finding a result equally or more
extreme if the null hypothesis is true
8. Compare the p-value to your stated rejection threshold




* significance level in hypothesis testing - probabiity we would be wrong when assuming the null
* P value -the probility of seeing something AS extreme OR MORE assuming the null hypothesis


* Type 1 error -  You reject your null hypothesis when you shouldn't have
* type 2 error - You fail to reject your null hypothesis when you should have



1. Can matt land a kick flip 80% of the time?
2. does he land the flip less than or equal to 80
3. Null-  less than or equal to 80.
4. 


at first the null would be p <= 0.8, but if you are more skeptical you wold say p =0.8 because that


p = 0.6

#### Review

* What is a random Varialbe- A numerical representation of a natural phenomona. 

* What is PMF - Discrete values
* Probability Density function - something will happend in a given time
* CDF = integral of the PDF, area under the curve depending what you're looking at. Typically, if looking for whats the probability between the bus arriving 20-30 minutes from now use pDF. Whats the probability of the bus arriving in at least 30 minutes use CDF
* 

-----------------------------------------

## Statistical Power

* Our ability to detect an affect when we dont have an affect

* ~/Desktop/den-19-dsi/lectures/statistical-power/alex-rose

* The power of our hypothesis test to detect an effect if there actually is one.
* Given that the true distribution is H1, the probability our test rejects the (false) null hypothesis.

* Significance level - a
* beta - b

* Higher alpha = higher type 1 error, higher power
* high effect size = higher power, lower type 2 error
* higher sample size = higher power, lower type 2 error


* Powers of hypothesis out comes --
    * correct - (1-a)
    * Type 1 error (False Positive) - (a)
    * Type 2 error (False negative) - (b)
    * correct - (1-b)

* Effect size - how far from the mean of the alternate hypothesis si away fromt he mean of the null




--------------------------------------------------------------------------------------

## Sampling Distributions

* Population : a set of similar items or events
    * Daily prices from the stock market
    * Possible customers of insurance company

* Sample : Subset of individuals from within a statsistal population
* Idendically distributed: sample items have the same distribution function

* The empirical distribution is 

## Sampling Theory

### BootStrap 
* [Boostrap example](http://localhost:8888/notebooks/sampling-distributions.ipynb)
* A bootstrap sample from a dataset is a sample taken with replacement from that dataset whose size is the size of the dataset itself.


* whats the point of bootstraping? - The Bootstrap is a tool to quantify the variation in a statistical estimate. It can be used in almost any situation. 

        * The bootstrap is a giant point in favor of the massive amount of computation all of us has at our disposal in modern day. Before the computer age, the practice of statistics was tedious and mathematical. Now we can estimate things earlier generations would never have dreamed of by simply putting to work some carefully engeneered slabs of silicon.

```python
def text_in_blank_plot(text, ax):
    '''make a text box'''
    _ = ax.text(0.5, 0.5, text, 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=15)
    ax.axis('off')


np.random.seed(123)
fig = plt.figure(figsize=(16, 4))
# colspan: Number of columns for the axis to span downwards.
ax = plt.subplot2grid((6, 3), (0, 0), colspan=2) 
ax.get_xaxis().set_ticks([])
ax.set_xlim(-2.5, 3)
one_dim_scatterplot(data, ax, s=15)

ax = plt.subplot2grid((6, 3), (0, 2), colspan=1)
text_in_blank_plot("Original Sample", ax)

# boostrapping 5 times
for i in range(0, 5):
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    ax = plt.subplot2grid((6, 3), (i + 1, 0), colspan=2)
    ax.get_xaxis().set_ticks([])
    ax.set_xlim(-2.5, 3)
    one_dim_scatterplot(bootstrap_sample, ax, c="black", s=15)
    sample_median = np.median(bootstrap_sample)
    ax.scatter([sample_median], 0, c="red", s=50)
    ax = plt.subplot2grid((6, 3), (i + 1, 2), colspan=1)
    text_in_blank_plot("Bootstrap Sample {}".format(i+1), ax)

```
```python
# function to define bootstrap medians
def bootstrap_sample_medians(data, n_bootstrap_samples=10**4):
    bootstrap_sample_medians = []
    for i in range(n_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_sample_medians.append(np.median(bootstrap_sample))
    return bootstrap_sample_medians

np.random.seed(321)
bootstrap_medians = bootstrap_sample_medians(data)

fig, ax = plt.subplots(1, figsize=(12, 4))
ax.hist(data, bins=25, density=True, color="black", alpha=0.4,
        label="Sample Data")
ax.hist(bootstrap_medians, bins=25, density=True, color="red", alpha=0.75,
        label="Bootstrap Sample medians")
ax.legend()
# ax.tick_params(axis='both', which='major', labelsize=15)
_ = ax.set_title("Bootstrap Sample medians (10000 samples)", fontsize = 20)



variance_of_sample = np.var(data)
varaince_of_bootstrap_medians = np.var(bootstrap_medians)

print("Variance of Sample: {:2.2f}".format(variance_of_sample))
print("Variance of Sample medians: {:2.2f}".format(varaince_of_bootstrap_medians))


```

```python
# 75 confidence interval with bootstrap

np.random.seed(333)
bootstrap_sample_75_percentiles = []
for i in range(10000):
    bootstrap = np.random.choice(data, size=len(data), replace=True)
    bootstrap_75_percentile = np.percentile(bootstrap, 75)
    bootstrap_sample_75_percentiles.append(bootstrap_75_percentile)

fig, ax = plt.subplots(1, figsize=(10, 4))
ax.hist(bootstrap_sample_75_percentiles, bins=500, density=True, color="black", alpha=0.5)
ax.set_title("boostrap sample 75 percentiles", fontsize=20)
# ax.tick_params(axis='both', which='major', labelsize=15)
left_endpoint = np.percentile(bootstrap_sample_75_percentiles, 2.5)
right_endpoint = np.percentile(bootstrap_sample_75_percentiles, 97.5)

print("Sample 75'th Percentile: {:2.2f}".format(np.percentile(data, 75)))
print("Bootstrap Confidence Interval for Population 75'th Percentile: [{:2.2f}, {:2.2f}]".format(
    left_endpoint, right_endpoint))
```


### Central Limit theorem

* [Central limit theorem lecture](http://localhost:8888/notebooks/pre-lecture-central-limit-theorem.ipynb)


*The stunning part of the central limit theorem is that it makes almost no assumptions about  𝑋 .  𝑋  can be anything, and it's sample means will always tend to be normal.

* The central limit theorem is a miracle, pure and simple. There is no real reason it is true, it just is. Consider it a gift of rare order in the universe, more like a fundamental law of physics than an intuitive mathematical fact.

* Here's an elevator pitch statement of the central limit theorem, good for job interviews: The central limit theorem allows us to make probabilistic statements about the sample mean from any population using the normal distribution.


```python
# how to find the midde 95% of the distribution
# how to plot vertical distribution lines
# how to find 95% confidence level


mu = 2 # mean
sigma = 0.5 #standard deviation

lunch = stats.norm(loc=mu, scale=sigma)

time_low = lunch.ppf(0.001)
time_high = lunch.ppf(0.999)
num_times = 100

times = np.linspace(time_low, time_high, num_times)
lunch_pdf = lunch.pdf(times)

time_025 = lunch.ppf(0.75)
time_975 = lunch.ppf(0.975)

fig, ax = plt.subplots(1, 1, figsize=(7,5))

ax.plot(times, lunch_pdf, c='k', label = 'distribution')
ax.axvline(time_025, color='red', linestyle='--', linewidth=1)
ax.axvline(time_975, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('time taken for lunch')
ax.set_ylabel('pdf')
ax.set_title('Middle 95% of distribution')
ax.legend()
plt.tight_layout()
plt.show()
```

```python

# how to create a shaded region below the curve

mask_gt = times >= time_025
mask_lt = times <= time_975              # these three create masks for the graph.
mask_middle = mask_gt & mask_lt

fig, ax = plt.subplots(1, 1, figsize=(7,5))

ax.plot(times, lunch_pdf, c='k', label = 'distribution')
ax.fill_between(times, lunch_pdf, 0, 
                where=mask_middle, color="red", alpha=0.2, label='middle 95%')
ax.set_xlabel('time taken for lunch')
ax.set_ylabel('pdf')
ax.set_title('Middle 95% - filled')
ax.legend(loc='upper right')
plt.tight_layout(w_pad=0.5)
plt.show()

```

```python
# plot 95% distribution

sample_mean = np.mean(data)
sample_varaince = np.var(data)
distribution_of_sample_minus_population_mean = stats.norm(0, np.sqrt(sample_varaince / len(data)))

fig, ax = plt.subplots(1, figsize=(10, 4))
x = np.linspace(-0.3, 0.3, num=250)
pdf = distribution_of_sample_minus_population_mean.pdf(x)
ax.plot(x, pdf, linewidth=3)

# Shade under curve
# Note: The 0.2 here is just for illustration, it does not correspond to
#       any particular value of alpha.

ax.set_xlim(-0.3, 0.3)
ax.fill_between(x, pdf, 0, 
                where=( (-0.2 <= x) * (x <= 0.2) ), color="red", alpha=0.2)
ax.text(-0.04, 1.5, "0.95", fontsize=35, color="red")
ax.set_xticks([]);

```

```python
# plot shaded area underneath distribution curve

fig, ax = plt.subplots(1, figsize=(10, 4))

ax.plot(x, pdf, linewidth=3)

# Shade under curve
ax.set_xlim(-0.3, 0.3)
ax.fill_between(x, pdf, 0, 
                where=( (-0.2 <= x) * (x <= 0.0) ), color="red", alpha=0.2)
ax.text(-0.15, 0.6, "0.475", fontsize=35, color="red")
ax.set_xticks([]);

```
```python
 # plot a 25% tail underneatht he distributon curve

 fig, ax = plt.subplots(1, figsize=(10, 4))

ax.plot(x, pdf, linewidth=3)

# Shade under curve
ax.set_xlim(-0.3, 0.3)
ax.fill_between(x, pdf, 0, 
                where=( (x <= -0.2) ), color="red", alpha=0.2)
ax.text(-0.18, 0.2, "0.025", fontsize=35, color="red")
ax.set_xticks([]);
```

#### Maximum likelihood

* The goal is to find the best distribution to fit the data
* [Maximum Likelihood jupyter notebook](http://localhost:8888/notebooks/maximum-likelihood.ipynb)


```python
# maximum likelihood function

def log_likelihood_normal_one_parameter(mu):
    normal = stats.norm(mu, 1,0)
    likelihoods = [normal.pdf(datum) for datum in data]
    return np.sum(np.log(likelihoods))

def log_likelihood_normal_two_parameters(mu, sigma_sq):
    normal = stats.norm(mu, np.sqrt(sigma_sq))
    likelihoods = [normal.pdf(datum) for datum in data]
    return np.sum(np.log(likelihoods))





def minus_log_likelihood_normal_two_parameters(mu, sigma):
    return -log_likelihood_normal_two_parameters(mu, sigma)

# The optimizer needs a function that consumes a single numpy array
def wrapper_for_scipy(x):
    return minus_log_likelihood_normal_two_parameters(x[0], x[1])
```

### Hypothesis testing

* [link to Jupyer notebook on hypothesis testing](http://localhost:8888/notebooks/hypothesis-testing.ipynb)



### Data vizulization lecture

* Types of data
    * Qualitative vs Quantitative
        *qualitative: data is descriptive information
        * Quantitative: Data is numerical information
    * Discrete vs Continuous
        * Discrete: Data can only take certain values
        * Continous: Data can take any value
    * Other types:
        * Nominal: Non numeric categories
        * Ordinal : Numeric data with non-constant or unknown spacing
        * Interval: Numeric data with uniform spacing
        * ratio: Interval data with a natural zero

* Scatter plots
    * data types: Continous, quantitative
    * Compaing an x variable to an Y variable
    * used to observe relationships between variables

* Line plots:
    * data types: continous, quantitive, interval, ordinal
    * Contstructed of lines connecting points called "markers"

* Histograms:
    * data types: Continous, Quantitiative
    * Create bins to separate the data, convention says that each bin is left inclusive, right exclusive
    * A comon function to calc number of bins is k = sqrt(n)
    * where k is the number of bins and n is the sample size
    


### Bayesian lecture

* Prior Probability - A PMF / PDF representing you intital beliefs about the parameter(s)
* Likelihood - The probability of observing the data given the parameter(s)
* Posterior propability - The product of prior and likelihood (bayesian-update)
                        - the posterior probability becomes the prior of the next update
* Normalizing constand - the probability of observing the data. 
* What is baye's rule
    * P(A|B) = P(B|A)P(A) / P(B)

```python

# function to flip a coin
def flip_the_coin(p, flips_lst):
    '''Flips the coin with probability of success p
       and appends to the flips_lst'''
    if np.random.random() <= p:
        flips_lst.append(1) #heads (success)
    else:
        flips_lst.append(0) #tails (failure)

# function to calculate likelihood / how to calculate likelihood

def calculate_likelihood(flips_lst):
    '''Likelihood of the last flip in the flips_lst given fair, not fair coin'''
    result = flips_lst[-1]
    likelihood_fair = stats.bernoulli.pmf(result, p_fair)
    likelihood_not_fair = stats.bernoulli.pmf(result, p_not_fair)
    return [likelihood_fair, likelihood_not_fair]

# how to calculate marginal / function to calculate marginal

def calculate_marginal(likelihoods_lst, prior_lst):
    prior_fair = prior_lst[0]
    prior_unfair = prior_lst[1]
    likelihood_fair = likelihoods_lst[0]
    likelihood_not_fair = likelihoods_lst[1]
    marginal = likelihood_fair * prior_fair + likelihood_not_fair * prior_unfair
    return marginal

# function to calculate posterior / how to calculate posterior

def calculate_posterior(likelihoods_lst, prior_lst):
    '''Calculates the posterior given the likelihoods and prior'''
    marginal = calculate_marginal(likelihoods_lst, prior_lst)
    posterior_unnormalized = []
    for likelihood, prior in zip(likelihoods_lst, prior_lst):
        posterior_unnormalized.append(likelihood * prior / marginal)
    # now need to normalize so that the total probability in posterior is 1
    posterior_un_total = sum(posterior_unnormalized)
    posterior_lst = []
    for posterior in posterior_unnormalized:
        posterior_lst.append(posterior/posterior_un_total)
    return posterior_lst

```

## **Types of Data**  

**Qualitative**: Descriptive information  
**Quantitative**: Numerical Information  
**Discrete**: Data can only take on certain values (int)  
**Continuous**: Data can take any value (float)  
**Nominal**: Non-numeric categories (brand)  
**Ordinal**: Numeric data with non-constant or unknown spacing (t-shirt sizes)  
**Interval**: Numeric data with uniform spacing (date)  
**Ratio**: Interval data with a natural zero  

![Graphing Order of Importance based on Data Type](images/data_visualization.png)

## **Types of Plots and when to use them:**  

**Scatter**:  
Data Types: Continuous, Quantitative  
Comparing an X variable to a Y variable  
Used to observe relationships between variables  

**Line Plots**:  
Data Types: Continuous, Quantitative, Interval, Ordinal  
Constructed of lines connecting points called “markers”  
X-axis is ordinal or interval in nature, like a time series  

**Histograms**:  
Data Types: Continuous, Quantitative  
Creates “bins” to separate the data, convention says that each ‘bin’ is left inclusive, right exclusive  
Can show the overall distribution of the data  
Right Skewed - tail goes to the right (left is opposite)  
To calculate the proper number of bins: `#_bins = sqrt(n_samples)`  

**Kernel Density Plot**  
Can show the overall distribution of the data on a continuous interval allowing for smoother distributions by smoothing the noise  

**Bar Chart**:  
Data Types: All  
Used to represent the same variable over a number of domains  
Can show frequency distributions for discrete variables  
Should be sorted in order if x-labels are not ordinal  

**Box Plot**:  
Data Types: Continuous, Quantitative  
Used to visually represent the five number summary  
Can show the distribution's skew  
Can visually represent outliers  

**Violin Plot**  
Similar to box plot, but adds in the kernel density plot in each side (shows the distribution shape)  
Shows the summary statistics in the plot as well  

**Pie Chart**  
Useless  

**Heat Maps**  
Visually represent a matrix  
Typically used to show covariance and correlation  

## One Dimensional Scatterplot
```python
def one_dim_scatterplot(data, ax, jitter=0.2, **options):
    ## why jitter? especially for bootstraping
    if jitter:
        jitter = np.random.uniform(-jitter, jitter, size=data.shape)
    else:
        jitter = np.repeat(0.0, len(data))
    ax.scatter(data, jitter, **options)
    ax.yaxis.set_ticklabels([])
    ax.set_ylim([-1, 1])
    ax.tick_params(axis='both', which='major', labelsize=15)

fig, ax = plt.subplots(1, figsize=(12, 1))
one_dim_scatterplot(data, ax, s=15)
```
![One Dimension Scatterplot](images/one_dimension_scatter_plot.png) 

## Empirical Distribution Plot

```python
def empirical_distribution(x, data):
    '''Cumulative distribution for the data'''
    weight = 1.0 / len(data)
    count = np.zeros(shape=len(x))
    for datum in data:
        count = count + np.array(x >= datum)
    return weight * count
```

# STATS and PROBABILITY

[Seeing Theory - Visualize Concepts](https://seeing-theory.brown.edu/basic-probability/index.html)  

## Docker lecture
* An open source project based on Linux containers
* A docker container is a stand-alone piece of doftware that includes everything needed to run it,

* Couple of commands
    * docker build
    * docker run
    * docker start
    * docker stop
    * docker pull
    * docker push
------------------------------------------------------------

## Sql Lecture
* RDBMS (relational Database Management system)
    * Schema - defines the structure of a table or database
    * Databaseis composed of a number of user-defined tables
    * Each table has columns and rows


* Sql Query Basics
    * SELECT - What data columns do you want
    * FROM - From what location do you want it
    * WHERE - What data (rows) do you want


* Order of operations sql
    * SELECT
    * FROM
    * JOIN / ON
    * WHERE
    * GROUP BY
    * HAVING
    * ORDER BY
    * LIMIT 
    * ;

# how to start up docker
$ docker start pgserv

$ docker exec -it pgserv bash
-# cd /home/data/path/to/data_dump/
-# psql -U postgres      <--- THIS ONE IS IMPORTANT
=# CREATE DATABASE new_database;
=# \q
-# psql -U postgres new_database < really_important.sql;
-# psql -U postgres new_database
=# \d
=# SELECT * FROM critical_table LIMIT 13;



### Some postgres commands
* \l - lists all databases
* \d - list all tables
* \d <tablename> - describe a table schema
* \h <clause> help for sql clause help
* q - exit current view and return to command line
* \q - quit sql
* \i script.sql - run script(or query)
------------------------------------------------------------------------


#### SQL python lecture

* General Workflow
1. Establish a connection to Postgres database using psycopg2
2. Create a cursor
3. Use the cursor to execute SQL queries and retrieve data
4. Commit SQL actions
5. Close the cursor and connection

```python
# Connect to the database
import psycopg2
import getpass

upass = getpass.getpass()
conn = psycopg2.connect(database="golf", user="ender", password=upass, host="localhost", port="5432")
print("connected")
```

```python 
# instantiate the cursor
cur = con.cursor()

# populate the database
run populate.py
```

```python
# look athe data one line at a time
cur.fetchone()

# many lines at a time
cur.fetchmany(5)

# or everything at once
resulults = cur.fetchall()

# how to iterate over a cursor
cur.execute(query)
for record in cur:
    print("data;{}, outlook:{}".format(record[0], record[1]))

# or another way
for row in cur
    print(row)

```

```python
# using pandas with sql
df = pd.read_sql(query, conn)
```

--------------------------------------------------------
### Algorithmic Time complexity lecture
* Big-o Notation: used to describe how the runtime (time complexity) and size (space complexity) of an algorithim increases as the size of the input array of the length N increases.

#### TIME complexity (best to worst)
* O(1) Constant - only a single step required to complete the task.
* O(log n) - Logarithmic - The number of steps it takes to accomplish the task are decreased by some factor with each step.
* O(n) - Linear - The number of steps required is directly related to n.
* O(n log n) - Log linear - The number of steps required is directly related to n multiplied by some factor that is a factor of n (but not much less than n)
* O(n^2) - Quadratic - The number of steps it takes to accomplish a task is square of n (BAD).
* O(C^N) - Exopential - The number of steps it takes to accomplish a task is a constant to the n power (VERY BAD)

#### Types of sorts
* Bubble sorts: large values bubble to the top. Pretty dang slow
```python
# psudo code for bubble sort
# O(n^2)

def bubbleSort(lst):
    for i in range(len(lst) -1, 1, -1):
        for j in range(j):
            if alist[j] > alist[j+1]:
                swap alist[j] and alist[j+1]
```                
* Insertion sort: The idea is that you keep one part of the last sorted as we go. O(n^2)
```python
# psudo code for insertion Sort
def insertionSort(lst):
    for i in range(1, len(lst)):
        val = lst[i]
        while i >0 and lst[i-1] > val:
            move lst[i-1] to the right
            decrement i
        assign val to lst[i]
```

----------------------------------------------

### Mongo DB lecture
* MongoDB is a document-oriented database, an alternative to RDBMS, used for storing semi-structured data.
* JSON like objects from the data model, rather than a RDBMS tabkes.
* Structure of the data base:
    * MongoDB is made up of databases which contain collections
    * A collection is made up of documents

* Querying data
```SQL
db.unicorns.find()
find is much more flexible.

// find by single field
db.unicorns.find({name: 'TwilightSparkle'})

// find by presence of field
db.unicorns.find({friends: {$exists : true}})

// find by value in array
db.unicorns.find({friends: 'TwilightSparkle'})

// To return only certain fields use the optional second argument to `find`.
// This says, return only the names of unicorns who are friends with
// twilight sparkle.
db.unicorns.find({friends: 'TwilightSparkle'}, {name: true})
```

* Updating data with mongo

```SQL
// Replaces friends array
db.unicorns.update({
    name: 'TwilightSparkle'}, {
    $set: {
        friends: ['Shutterfly', 'Rarity', 'Applejack']}})

// Adds to friends array
db.unicorns.update({
    name: 'Applejack'}, {
    $push: {
        friends: 'Rarity'}})
We have to use the $set and $push operators, the default behaviour of update is to replace the data.

// Replaces the TwighlightSparkle data completely!
// It will no longer have even a name field after this!
db.unicorns.update({
    name: 'TwilightSparkle'}, {
    friends: ['Shutterfly', 'Rarity', 'Applejack']})
An upsert either creates a document (when it does not already exist) or inserts into an existing document.

// Upsert: This one is created
db.unicorns.update({
    name: "Rarity"}, {
    $push: {
        friends: {
            $each: ["TwilightSparkle", "Applejack", "Fluttershy"]}}}, {
    upsert: true})

// Upsert: This one is updated
db.unicorns.update({
    name: "Fluttershy"}, {
    $push: {
        friends: {
            $each: ["Rarity", "PrincessCelestia"]}}}, {
    upsert: true})
```
* Upsert in mongo
    * if the record exists, were going to update the record with new info. If it dosent exist, its going to create a new record with the new provided info

* how to remove data in mongo
```SQL
db.unicorns.remove({name: "Fluttershy"})
```

-------------------------------------------

### Web scraping lecture

* important html tags

```html
<a href="http://www.w3schools.com">A hyperlink to W3Schools.com!</a>

<h1>This is a header!</h1>

<p>This is a paragraph!</p>

<h2>This is a Subheading!</h2>

<table>
  This is a table!
  <tr>
    <th>The header in the first row.</th>
    <th>Another header in the first row.</th>
  </tr>
  <tr>
    <td>An entry in the second row.</td>
    <td>Another entry in the second row.</td>
  </tr>
</table>

<ul>
  This is an unordered list!
  <li>This is the first thing in the list!</li>
  <li>This is the second thing in the list!</li>
</ul>
<div>Specifies a division of the document, generally with additional attributes specifying layout and behavior.</div>
A <span>span is similar</span> but occurs in the middle of a line.
```
* Web vs Internet : Internet came first. 
    * Internet includes things like : email, SSH. Really just a series of protocols

* HTTP Requests
    * To get data from the web, you need to make a HTTP request. The two most important request types are:

* GET (queries data, no data is sent)
* POST (updates data, data must be sent)



#### Steps when scraping a website

1. Check out the website in a browser
    * right click inspect, find the right tags

2. Send a GET request for the data

    ```python
     deer_tier_url = 'http://deertier.com/Leaderboard/AnyPercentRealTime'
    r = requests.get(deer_tier_url)
    r.status_code
    r.content
    ```
3. Save all the hypter text into mongo for later use

    ```python
    client = MongoClient('localhost', 27017)
    db = client.metroid
    pages = db.pages

    pages.insert_one({'html': r.content})
    ```

4. Parse the hypertext in BeautifulSoup

    ```python
    soup = BeautifulSoup(r.content, 'html')
    print(soup)
    print(soup.prettify())
    print(soup.title)
5. navigate the data to pull out the table information you want

```python
div = soup.find("div", {"class": "scoreTable"})
table = div.find("table")

# This returns an iterator over the rows in the table.
rows = table.find_all("tr")

all_rows = []

# Let's store each row as a dictionary 
empty_row = {
    "rank": None, "player": None, "time": None, "comment": None
}

# The first row contains header information, so we are skipping it.
for row in rows[1:]:
    new_row = copy.copy(empty_row)
    # A list of all the entries in the row.
    columns = row.find_all("td")
    new_row['rank'] = int(columns[0].text.strip())
    new_row['player'] = columns[1].text.strip()
    new_row['time'] = columns[2].text.strip()
    new_row['comment'] = columns[4].text.strip()
    all_rows.append(new_row)    

pprint.pprint(all_rows[:4])
```

6. Load all the rows into a mongo database

```python
db = client.metroid
deer_tier = db.deer_tier
for row in all_rows:
    deer_tier.insert_one(row)
```
7. Load all the rows into a pandas dataframe

```python
rows = deer_tier.find()
super_metroid_times = pd.DataFrame(list(rows))

uper_metroid_times.head()

super_metroid_times = super_metroid_times.drop("_id", axis=1)
super_metroid_times = super_metroid_times.set_index("rank")
super_metroid_times.head()
```
-------------------------------------------

#### How to use an API to scrape (wikipedia page)

```python
import json
import re

# The `User-Agent` section of a HTTP header contains this information.Wikipedia wants us to identify ourselves before it will give us data

headers = {'User-Agent': 'GalvanizeDataWrangling/1.1 matthew.drury@galvanize.com'}

api_url = 'https://en.wikipedia.org/w/api.php'

# Parameters for the API request: We want the Unicorn page encoded as json.
payload = {'action': 'parse', 'format': 'json', 'page': "Unicorn"}

r = requests.post(api_url, data=payload, headers=headers)

print(r.json().keys())
```

```python
# STEP 2: STORE THE DATA IN MONGODB
# import MongoDB modules
from pymongo import MongoClient
from bson.objectid import ObjectId

# connect to the hosted MongoDB instance
client = MongoClient('localhost', 27017)
db = client.wikipedia

collection = db.wikipedia

if not collection.find_one(r.json()['parse']):
    collection.insert_one(r.json()['parse'])

unicorn_article = collection.find_one({ "title" : "Unicorn"})

pprint.pprint(unicorn_article)

print (unicorn_article.keys())
```

```python
# STEP 3: Retrieve and store every article with associated metadata within on link

links = unicorn_article['links']

pprint.pprint(links)

len(links)

for link in links:

    payload = {'action': 'parse' ,'format': 'json', 'page' : link['*'] }
    r = requests.post(api_url, data=payload, headers=headers)

    # check to first see if the document is already in our database, if not, store it.
    try:
        j = r.json()
        if not collection.find_one(j['parse']):
            print("Writing The Article: {}".format(j['parse']['title']))
            collection.insert_one(j['parse'])
    except Exception as e:
        print(e)
```

```python
# Find all articles that mention 'Horn' or 'Horned'

# compile our regular expression since we will use it many times
regex = re.compile(' Horn | Horned ', re.IGNORECASE)

with open('wiki_articles.txt', 'w') as out:

    for doc in collection.find():
        
        # Extract the HTML from the document
        html = doc['text']['*']

        # Stringify the ID for serialization to our text file
        doc['_id'] = str(doc['_id'])

        # Create a Beautiful Soup object from the HTML
        soup = BeautifulSoup(html)

        # Extract all the relevant text of the web page: strips out tags and head/meta content
        text = soup.get_text()

        # Perform a regex search with the expression we compiled earlier
        match = regex.search(text)

        # if our search returned an object (it matched the regex), write the document to our output file
        if match:
            try:
                print("Writing Article: {}".format(doc['title']))
                json.dump(doc, out) 
                out.write('\n')
            except UnicodeEncodeError as e:
                print(e)

    out.close()
```

# Master DSI notes


 ## Relevant links 
* [Course Page](https://github.com/GalvanizeDataScience/course-outline/tree/20-10-DS-DEN_DEN19)

* [Quick Reference guide](https://github.com/GalvanizeDataScience/course-outline/tree/20-10-DS-DEN_DEN19/quick-reference)

* [Pandas Groupby syntax](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html)

* [Matplotlib intro syntax]()


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




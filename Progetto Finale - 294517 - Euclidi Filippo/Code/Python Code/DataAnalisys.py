# @Author Euclidi Filippo - matr. 294517

# Plot figures might be a little bigger than supposed to, this code was used in a notebook 
# environment, so data graphs size might be different in python

#Importing all needed libraries.
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#Setting train and test set by importing the datasets
df_train = pd.read_csv("./Code/Python Code/archive/train.csv")
df_test = pd.read_csv("./Code/Python Code/archive/test.csv")

# Checking if any of the values in our dataframe are null

check_any_NaN_Test = df_test.isnull().values.any()
check_any_NaN_Train = df_train.isnull().values.any()
print("Any NaN values in the train dataframe: " + str(check_any_NaN_Train) + 
      " and in the test dataframe: " + str(check_any_NaN_Test))

# We can be very happy since there are no null or NaN values 
# Lets now visualize the first 5 rows of our database to have a look at the
# columns and the type of data contained in them

print(df_train.head())

# Lets do the same for the test set

print(df_test.head())

# Now lets see the shape of our dataset to understan the numbers we are working with

print(df_train.shape)
print(df_test.shape)

# As we can see al data is numerical, and most of it is quantitative.
# Let's see some statistics of some of these columns

# First lets see the number of phones for each price rance
print(df_train['price_range'].value_counts())

# We can confidently say that our dataset is perfectly balanced
# Lets do some basic statistical analysis on the datasets, by using the function dataframe.describe()

print(df_test.describe())

# Lets now create a function to visualize the data by plotting it in some histogram
# graphs and learn more about the data contained in each column.
# This function was inspired by an already existing function and was modified for my needs

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape #number of columns and rows to shape the hist.
    columnNames = list(df) #getting the labels for each column
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow #number of rows of the graph
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k') #creating the plot and it's design
    for i in range(min(nCol, nGraphShown)): 
        plt.subplot(int(nGraphRow), int(nGraphPerRow), i + 1) #casting to int for deprecation warning by matplotlib
        columnDf = df.iloc[:, i] 
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})') 
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    
print(plotPerColumnDistribution(df_train, 20, 5))


df_train['screen_resolution'] = (df_train['px_height']*df_train['px_width'])

print(df_train.head())

# As we can see some columns contain Y/N values for specific components of the
# phone, so we can say that these columns contain only categorical values.
# Specifically these are the bluetooth column, the dual sim column, the 4G column, 
# the 3G column, the Touch Screen column and the wifi column.
# Also the price range columns contains some categorical values.

# Now lets use the Pearson Correlation Matrix to visualize the correlation of 
# the data in the training dataset 

correlation_df_train = df_train.corr(method = 'pearson')
plt.figure(figsize=(18, 18), facecolor='w')
sns.heatmap(correlation_df_train, annot=True)
plt.title(f'Correlation Matrix for Train Dataset', fontsize=15)
plt.show()

# As we can see both the training and test set are very similar and there is a good
# correlation between the price and the ram, some correlation between price and battery 
# power and a minor correlation between price and screen resolution (pixel in width and height)

# Lets see this correlation in a regplot with Seaborn

print("Regplot for price range and ram")
regplot1 = sns.regplot(x = "ram", y = "price_range", data=df_train)

# There is a good correlation between ram and price range as shown in the regplot

print("Barplot for price range and screen resolution")
sns.barplot(y = "screen_resolution", x = "price_range", data=df_train)

# There is a poor correlation between the screen resolution and price range, 
# especially between the medium and high range we can say that screen resolution isn't 
# correlated to the price

print("Barplot for price range and battery power")
sns.barplot(y = "battery_power", x = "price_range", data=df_train)

# It's possible to see that the correlation between ram and price range is actually effective
# and this will be one of the best variables, while the screen resolution is not as good
# some of the data could also be imprecise as some pixels resolutions are very uncommon
# we will still keep it as we cant be sure that those screens don't actually exist.

# We will now group all the categorical variables in a sub set (except for the price range)

categorical_var = ['blue', 'dual_sim', 'three_g', 'four_g', 'touch_screen', 'wifi'] 

# Also as 0, 1, 2 and 3 are not very meaningful we as price range, we can change them to a more
# meaningful low, medium, high, very high, which are the most used words to place a phone in a 
# specific price range, which will made our following analysis easier to understand.

df_train['price_range_qualitative'] = df_train['price_range'].replace(to_replace=[0,1,2,3],
                                    value=['low', 'medium', 'high', 'very high'])

print(df_train.head()) #lets visualize it to see if everything is correct

# We are now going to group the categorical variables by price range and see if 
# some of them are linked to the price somehow.
# If we see that there is some correlation between a phone not having (or having) a specific 
# component and it being in a lower (or higher) price category, then it means that the specific 
# variable could be in some way connected to the price range of that phone.

print(df_train.groupby([categorical_var[0]])['price_range_qualitative'].value_counts(normalize=True))
print(df_train.groupby([categorical_var[1]])['price_range_qualitative'].value_counts(normalize=True))
print(df_train.groupby([categorical_var[2]])['price_range_qualitative'].value_counts(normalize=True))
print(df_train.groupby([categorical_var[3]])['price_range_qualitative'].value_counts(normalize=True))
print(df_train.groupby([categorical_var[4]])['price_range_qualitative'].value_counts(normalize=True))
print(df_train.groupby([categorical_var[5]])['price_range_qualitative'].value_counts(normalize=True))


# To see the correlation more clearly, let's put everything in some Facetgrid graphs, as x 
# axis we are going to use the ram as we have seen that it shows a good correlation to the price
# and on the y axis we will use the selected categorical value, for the hue we will use 
# the price range

bins_size = np.linspace(df_train.ram.min(), df_train.ram.max(), 10)
b = sns.FacetGrid(df_train, hue = "price_range", col = "blue")
b.map(plt.hist, 'ram', bins = bins_size, ec = 'k')
b.add_legend()

s = sns.FacetGrid(df_train, hue = "price_range", col = "dual_sim")
s.map(plt.hist, 'ram', bins = bins_size, ec = 'k')
s.add_legend()

g3 = sns.FacetGrid(df_train, hue = "price_range", col = "three_g")
g3.map(plt.hist, 'ram', bins = bins_size, ec = 'k')
g3.add_legend()

g4 = sns.FacetGrid(df_train, hue = "price_range", col = "four_g")
g4.map(plt.hist, 'ram', bins = bins_size, ec = 'k')
g4.add_legend()

ts = sns.FacetGrid(df_train, hue = "price_range", col = "touch_screen")
ts.map(plt.hist, 'ram', bins = bins_size, ec = 'k')
ts.add_legend()

w = sns.FacetGrid(df_train, hue = "price_range", col = "wifi")
w.map(plt.hist, 'ram', bins = bins_size, ec = 'k')
w.add_legend()

plt.show()

# As we can see only the 3G has a good correlation to the price range while all
# the others have mixed results which show no statistical correlation

# We have now finished the part of data analysis and data visualization
# and we will now pass to the pre-processing and the machine learning model implementation
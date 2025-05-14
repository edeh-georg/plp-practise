# Step 1: Load the Dataset
import pandas as pd

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, header=None, names=columns)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_data.head())

# Step 2: Explore the Structure of the Dataset
print("\nData types and missing values:")
print(iris_data.info())
print(iris_data.isnull().sum())

# Step 3: Clean the Dataset
iris_data.dropna(inplace=True)  # Uncomment if needed

# Task 2: Basic Data Analysis

# Step 1: Compute Basic Statistics
print("\nBasic statistics of the numerical columns:")
statistics = iris_data.describe()
print(statistics)

# Step 2: Grouping and Analysis
print("\nMean of numerical columns grouped by species:")
grouped_data = iris_data.groupby('species').mean()
print(grouped_data)


# Task 3: Data Visualization

# Importing matplotlib for visualization
import matplotlib.pyplot as plt

# Step 1: Bar Chart
# Bar chart for average petal length per species
plt.figure(figsize=(10, 6))
grouped_data['petal_length'].plot(kind='bar', color='skyblue')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Step 2: Histogram
# Histogram of petal length
plt.figure(figsize=(10, 6))
plt.hist(iris_data['petal_length'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Step 3: Scatter Plot
# Scatter plot of sepal length vs petal length
plt.figure(figsize=(10, 6))
plt.scatter(iris_data['sepal_length'], iris_data['petal_length'], c=iris_data['species'].astype('category').cat.codes, cmap='viridis', alpha=0.7)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.colorbar(ticks=[0, 1, 2], label='Species', format='%d')
plt.grid()
plt.show()

# Conclusion
# In this analysis, we explored the Iris dataset, computed basic statistics, and visualized the data using various plots.
# The average petal length and width vary significantly across species, indicating distinct characteristics among them.

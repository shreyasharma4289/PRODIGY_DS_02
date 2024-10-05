import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:/Users/Shreya Sharma/Downloads/Titanic-Dataset.csv")

# View the first few rows
print("First five rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# --- Data Cleaning ---
# Fill missing 'Age' values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the most frequent value (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Check again for missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# --- Feature Engineering ---
# Create a new feature 'FamilySize' by combining 'SibSp' and 'Parch'
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Create 'AgeGroup' feature by categorizing 'Age' into bins
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child', 'Teenager', 'Adult', 'Middle-Aged', 'Senior'])

# View the modified dataframe with new features
print("\nFirst five rows of the dataset after feature engineering:")
print(df.head())

# --- Exploratory Data Analysis (EDA) ---
# a. Overview of the dataset
print("\nBasic statistics of the dataset:")
print(df.describe())

# b. Survival count plot
print("\nPlotting survival count...")
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()  # Display the plot

# c. Impact of Gender on Survival
print("\nPlotting survival count by gender...")
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.show()

# d. Impact of Passenger Class on Survival
print("\nPlotting survival count by passenger class...")
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Count by Passenger Class')
plt.show()

# e. Age distribution by Survival
print("\nPlotting age distribution by survival...")
sns.histplot(df[df['Survived'] == 1]['Age'], bins=20, label='Survived', kde=False, color='green')
sns.histplot(df[df['Survived'] == 0]['Age'], bins=20, label='Not Survived', kde=False, color='red')
plt.title('Age Distribution by Survival')
plt.legend()
plt.show()

# f. Impact of Embarked Port on Survival
print("\nPlotting survival count by embarkation port...")
sns.countplot(x='Survived', hue='Embarked', data=df)
plt.title('Survival Count by Embarkation Port')
plt.show()

# --- Correlation Analysis ---
print("\nCalculating correlation matrix...")
corr_matrix = df.corr()

print("\nCorrelation matrix:")
print(corr_matrix)

# Plotting heatmap of the correlation matrix
print("\nPlotting correlation heatmap...")
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

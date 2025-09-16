import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset from seaborn
df = sns.load_dataset('titanic')

# Preview data
print("Initial data preview:")
print(df.head())
print("\nData info:")
print(df.info())

# Data Cleaning
# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing 'age' with median
df['age'].fillna(df['age'].median(), inplace=True)

# Fill missing 'embarked' with mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop 'deck' column (too many missing values)
df.drop(columns=['deck'], inplace=True)

# Drop rows with missing 'embark_town' (small number)
df.dropna(subset=['embark_town'], inplace=True)

# Verify no missing values now
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Basic EDA

# 1. Distribution of survival
plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=df)
plt.title('Survival Counts (0 = No, 1 = Yes)')
plt.savefig('survival_counts.png')
plt.show()

# 2. Survival rate by gender
plt.figure(figsize=(6,4))
sns.barplot(x='sex', y='survived', data=df)
plt.title('Survival Rate by Gender')
plt.savefig('survival_by_gender.png')
plt.show()

# 3. Age distribution by survival
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='age', hue='survived', bins=30, kde=True, multiple='stack')
plt.title('Age Distribution by Survival')
plt.savefig('age_distribution_by_survival.png')
plt.show()

# 4. Survival rate by passenger class
plt.figure(figsize=(6,4))
sns.barplot(x='pclass', y='survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.savefig('survival_by_pclass.png')
plt.show()

# 5. Correlation heatmap of numerical variables
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Save cleaned data for reference
df.to_csv('titanic_cleaned.csv', index=False)

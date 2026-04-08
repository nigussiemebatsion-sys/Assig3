import pandas as pd

# Part 1: my own dataset
data = {
    "Name": ["Aman","Sara","John","Liya","Noah","Marta","Eden","Abel","Ruth","Sam",
             "Helen","Mark","Lily","Tom","Anna"],
    "Age": [20,22,23,21,24,22,23,25,21,22,23,24,20,21,22],
    "Gender": ["M","F","M","F","M","F","F","M","F","M","F","M","F","M","F"],
    "Score": [85,90,78,88,92,95,87,76,89,84,91,77,93,80,86],
    "Passed": ["Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes"]
}

index_labels = [f"ID_{i}" for i in range(1,16)]
df1 = pd.DataFrame(data, index=index_labels)
print("Part 1 dataset")
print(df1)

# Part 2: Titanic dataset
file_path = r"C:\Users\Administrator\Downloads\Titanic-Dataset.csv"
titanic = pd.read_csv(file_path)

# exploring the data
print("first few rows")
print(titanic.head())

print("info about data")
print(titanic.info())

print("some statistics")
print(titanic.describe())

# cleaning the data
# fix missing ages with median
titanic["Age"].fillna(titanic["Age"].median(), inplace=True)

# fix missing embarked with most common value
titanic["Embarked"].fillna(titanic["Embarked"].mode()[0], inplace=True)

# cabin has too many missing so just remove
if "Cabin" in titanic.columns:
    titanic.drop(columns=["Cabin"], inplace=True)

# remove duplicates if any
titanic.drop_duplicates(inplace=True)

# analysis
print("survival rate by gender")
print(titanic.groupby("Sex")["Survived"].mean())

print("survival rate by class")
print(titanic.groupby("Pclass")["Survived"].mean())

print("average age per class")
print(titanic.groupby("Pclass")["Age"].mean())

# make age groups
titanic["AgeGroup"] = pd.cut(titanic["Age"], bins=[0,12,18,35,60,100],
                             labels=["Child","Teen","Young Adult","Adult","Senior"])

print("survival rate by age group")
print(titanic.groupby("AgeGroup")["Survived"].mean())

# filtering
print("female passengers who survived")
print(titanic[(titanic["Sex"]=="female") & (titanic["Survived"]==1)])

print("children who survived")
print(titanic[(titanic["Age"]<12) & (titanic["Survived"]==1)])

print("1st class passengers who survived")
print(titanic[(titanic["Pclass"]==1) & (titanic["Survived"]==1)])
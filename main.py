# Drawing conclusions with data visualization

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("Expanded_data_with_more_features.csv")
df.head()
df.shape
df.describe()
df.info()

df.isnull().sum()

# Drop unnamed column
df = df.drop("Unnamed: 0", axis=1)
df.head()

# Change weekly study hours columns(5-10 inplace 5-Oct)
df["WklyStudyHours"] = df["WklyStudyHours"].str.replace("05-Oct", "5-10")

# Gender distribution
df["Gender"].value_counts()

plt.figure(figsize=(5, 5))
ax = sns.countplot(df, x="Gender")
ax.bar_label(ax.containers[0])  # to show the exact count value
plt.title("Gender Distribution")
plt.show()
# From the chat above we analysed that:
# The number of female students is more than male students


gb = df.groupby("ParentEduc").agg({"MathScore": "mean", "ReadingScore": "mean", "WritingScore": "mean"})
plt.figure(figsize=(12, 5))
sns.heatmap(gb, annot=True)
plt.title("Relationship between parent education and student score")
plt.show()
# From the above chart we analyzed that the education of parents has a good impact on students' scores


gb1 = df.groupby("ParentMaritalStatus").agg({"MathScore": "mean", "ReadingScore": "mean", "WritingScore": "mean"})
plt.figure(figsize=(12, 5))
sns.heatmap(gb1, annot=True)
plt.title("Relationship between parent marital statue and student score")
plt.show()
# From the above chart wee analyzed that the marital statue of parents
# has no/negligible impact on students' score


df.head()
gb2 = (df.groupby(["ParentMaritalStatus", "IsFirstChild", "NrSiblings"])
       .agg({"MathScore": "mean", "ReadingScore": "mean", "WritingScore": "mean"}))
plt.figure(figsize=(25, 25))
sns.heatmap(gb2, annot=True)
plt.title("Impact of family members on student score")
plt.show()
# From the above chart we conclude that the widowed parent with >5 siblings
# has negative impact if the student is not the first child


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(df, x="MathScore")

plt.subplot(1, 3, 2)
sns.boxplot(df, x="WritingScore")

plt.subplot(1, 3, 3)
sns.boxplot(df, x="ReadingScore")

plt.tight_layout()
plt.show()
# From the graph above we can say that students are weak in math
# comparing with reading and writing


# Distribution of ethnic groups
df["EthnicGroup"].unique()
df["EthnicGroup"].nunique()
groupA = df.loc[(df['EthnicGroup'] == "group A")].count()
groupB = df.loc[(df['EthnicGroup'] == "group B")].count()
groupC = df.loc[(df['EthnicGroup'] == "group C")].count()
groupD = df.loc[(df['EthnicGroup'] == "group D")].count()
groupE = df.loc[(df['EthnicGroup'] == "group E")].count()

l1 = ["group A", "group B", "group C", "group D", "group E"]
mlist = [groupA["EthnicGroup"], groupB["EthnicGroup"], groupC["EthnicGroup"],
         groupD["EthnicGroup"], groupE["EthnicGroup"]]
plt.pie(mlist, labels=l1, autopct="%1.2f%%")
plt.title("Distribution of Ethnic Groups")
plt.show()
# From the above chart we conclude that most(31%) of the students are in group C

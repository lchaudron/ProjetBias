import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/brute/data.csv', sep=';')


df['Target'] = np.where(df['Target'] == 'Dropout', 1, 0)

print(df['Target'].value_counts())

plt.figure(figsize=(6, 4))
df['Age at enrollment'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Age at Enrollment')
plt.xlabel('Age at Enrollment')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('outputs/viz/prepro/age_at_enrollment_distribution.png')
#plt.show()

plt.figure(figsize=(6, 4))
df['Previous qualification'].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Distribution of Previous Qualification')
plt.xlabel('Previous Qualification')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('outputs/viz/prepro/previous_qualification_distribution.png')
#plt.show()

plt.figure(figsize=(6, 4))
df['Course'].value_counts().plot(kind='bar', color='lightgreen', edgecolor  ='black')
plt.title('Distribution of Course')
plt.xlabel('Course')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('outputs/viz/prepro/course_distribution.png')
#plt.show()

plt.figure(figsize=(6, 4))
df['Marital status'].value_counts().plot(kind='bar', color='lightgreen', edgecolor  ='black')
plt.title('Distribution of Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('outputs/viz/prepro/marital_status_distribution.png')
#plt.show()

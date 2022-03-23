import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

df = pd.read_csv('Employee_Data.csv')
sns.set(style="darkgrid")

# distribution of age

std = np.std(df['Age'], ddof=1)
std = round(std,2)
mean = np.mean(df['Age'])
mean = round(mean,2)
domain = np.linspace(np.min(df['Age']), np.max(df['Age']))
plt.plot(domain, norm.pdf(domain, mean, std),label='$\mathcal{N}$' + f'$( \mu\\approx{(mean)}, \sigma\\approx{(std)})$')
plt.hist(df['Age'], edgecolor='black', alpha=.5, density=True)
plt.legend()
plt.title("Normal Distribution for Age of employees")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# distribution of expertise

std = np.std(df['Expertise'], ddof=1)
std = round(std,2)
mean = np.mean(df['Expertise'])
mean = round(mean,2)
domain = np.linspace(np.min(df['Expertise']), np.max(df['Expertise']))
plt.plot(domain, norm.pdf(domain, mean, std),label='$\mathcal{N}$' + f'$( \mu\\approx{(mean)}, \sigma\\approx{(std)})$')
plt.hist(df['Expertise'], edgecolor='black', alpha=.5, density=True)
plt.legend()
plt.title("Normal Distribution for Expertise level of employees")
plt.xlabel("Expertise")
plt.ylabel("Frequency")
plt.show()

# distribution of yrs of exp

std = np.std(df['Yrs of Experience'], ddof=1)
std = round(std,2)
mean = np.mean(df['Yrs of Experience'])
mean = round(mean,2)
domain = np.linspace(np.min(df['Yrs of Experience']), np.max(df['Yrs of Experience']))
plt.plot(domain, norm.pdf(domain, mean, std),label='$\mathcal{N}$' + f'$( \mu\\approx{(mean)}, \sigma\\approx{(std)})$')
plt.hist(df['Yrs of Experience'], edgecolor='black', alpha=.5, density=True)
plt.legend()
plt.title("Normal Distribution for Years of Experience of employees")
plt.xlabel("Years of Experience")
plt.ylabel("Frequency")
plt.show()

# grp count plot

ax = sns.countplot(x='Group',data=df)
plt.show()

# Age and Expertise level cluster

X = df.iloc[:, [1, 9]].values
x = df['Age']
y = df['Expertise']

# view data
plt.title("K- Means for Employee Details")
plt.xlabel("Age")
plt.ylabel("Expertise")
plt.scatter(x, y, label='Training data')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
plt.show()

#wcss graph

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, label='Number of clusters')
plt.title("K- Means for Employee Details")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
plt.show()

# final graph
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Assembler & Machinist/Operator')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Quality control Inspector')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Production Manager')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='black',label='Centroids')
plt.title("K- Means for Employee Details")
plt.xlabel("Age")
plt.ylabel("Expertise")
plt.legend()
plt.show()

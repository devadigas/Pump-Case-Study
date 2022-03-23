
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('Annual_Data.csv')

sns.set(style="darkgrid")

# Years vs no of employees
x = df['Year']
y = df['Employees']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Number of Employees')
plt.show()

# Years vs working days
x = df['Year']
y = df['Working Days']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Working Days')
plt.show()

# Years vs raw material
x = df['Year']
y = df['Raw material(in million)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Cost of Raw Material(in million)')
plt.show()

# Years vs machinery cost
x = df['Year']
y = df['Machinery cost(in million)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Cost of Machinery(in millions)')
plt.show()

# Years vs overhead cost
x = df['Year']
y = df['Overhead cost(in million)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Overhead Cost(in million)')
plt.show()

# Years vs labor cost
x = df['Year']
y = df['Labor cost(in million)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Labor cost(in million)')
plt.show()

# Years vs profit
x = df['Year']
y = df['Profit(in millions)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Profit earned(in million)')
plt.show()


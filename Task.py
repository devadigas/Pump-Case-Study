import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Assembler.csv')
data = pd.read_csv('Machinist.csv')
dataset = pd.read_csv('Production_Manager.csv')
d1 = pd.read_csv('Quality_Control.csv')

sns.set(style="darkgrid")
"""
# For assembler
# years vs employees

x = df['Year']
y = df['Employees']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Number of Employees')
plt.show()

# years vs Automation

x = df['Year']
y = df['Automation(in %)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Automation')
plt.show()

# years vs time spent

x = df['Year']
y = df['Time spent(Out of 40)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Time spent')
plt.show()

# For machinist
# years vs employees

x = data['Year']
y = data['Employees']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Number of Employees')
plt.show()

# years vs Automation

x = data['Year']
y = data['Automation(in %)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Automation')
plt.show()

# years vs time spent

x = data['Year']
y = data['Time spent(out of 30)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Time spent')
plt.show()

# For production manager
# years vs employees

x = dataset['Year']
y = dataset['Employees']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Number of Employees')
plt.show()

# years vs Automation

x = dataset['Year']
y = dataset['Automation(in %)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Automation')
plt.show()

# years vs time spent

x = dataset['Year']
y = dataset['Time Spent(out of 15)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Time spent')
plt.show()

# For quality
# years vs employees

x = d1['Year'].astype(str)
y = d1['Employees']
plt.plot(x, y)
plt.xlabel('Year')
plt.ylabel('Number of Employees')
plt.xticks(rotation=50)
plt.show()
"""
# years vs Automation

x = d1['Year'].astype(str)
y = d1['Automation(in %)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Automation')
plt.xticks(rotation=50)
plt.show()

# years vs time spent

x = d1['Year'].astype(str)
y = d1['Time spent(out of 15)']
plt.plot(x,y)
plt.xlabel('Year')
plt.ylabel('Time spent')
plt.xticks(rotation=50)
plt.show()

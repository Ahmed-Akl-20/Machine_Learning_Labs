from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1],[2],[3],[4],[5]])
y = np.array([60,70,80,85,90])

model = LinearRegression()
model.fit(x,y)

x_new = np.array([[6]])
y_pred = model.predict(x_new)
print(f"Pridected Grade For 6 Hours : {y_pred[0]:2f}")

plt.scatter(x, y, color='blue' , label='Data')
plt.plot(x, model.predict(x),  color='red' , label='Fit Line')
plt.xlabel('study Hours')
plt.ylabel('Grades')
plt.title('Linear Regression')
plt.legend()
plt.show()
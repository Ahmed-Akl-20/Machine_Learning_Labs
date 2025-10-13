from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,1,1,1])

model = LogisticRegression()
model.fit(x,y)

x_new = np.array([[3.5]])
y_pred = model.predict(x_new)
print(f"Pridection For 3.5 Hours : {'pass' if y_pred[0] == 1 else 'Fail'}")

x_plot = np.linspace(0,6,100).reshape(-1,1)
y_prob = model.predict_proba(x_plot)[:,1]

plt.scatter(x, y, color='blue' , label='Data')
plt.plot(x_plot, y_prob,  color='red' , label='Pass Probability')
plt.xlabel('study Hours')
plt.ylabel('Pass Probability')
plt.title('Logistec Regression')
plt.legend()
plt.show()
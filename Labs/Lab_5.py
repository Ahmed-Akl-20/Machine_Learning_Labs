import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = {
    'Hours_Study' : [2,3,4,6,8],
    'Sleep_Hours' : [9,6,8,5,3],
    'Result' : [0,0,1,1,1]
}
df = pd.DataFrame(data)

X = df[['Hours_Study' , 'Sleep_Hours']]
y = df['Result']

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)

# test = [[5,7]]
test = [[2,9]]
prediction = knn.predict(test)
print("Predicted Class  (1=Pass , 0=Fail) : ",prediction)
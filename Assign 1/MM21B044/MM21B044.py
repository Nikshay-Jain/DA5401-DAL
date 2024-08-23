import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def display(mat):
    coordinates = np.argwhere(mat == 1)
    rows, cols = coordinates[:, 0], coordinates[:, 1]
    plt.scatter(rows,cols)
    plt.show()

data = pd.read_csv("Default Dataset.csv")
row0 = np.array((data.columns[0],data.columns[1]), dtype=np.float64)
data = data.rename(columns={data.columns[0]: 'X', data.columns[1]: 'Y'})
data = pd.concat([pd.DataFrame([row0], columns=data.columns), data], ignore_index=True)

X = (np.round(data["X"].to_numpy()*10)).astype(int)
Y = (np.round(data["Y"].to_numpy()*10)).astype(int)
plt.scatter(X,Y)
plt.show()

mat = np.zeros((1000,1000))
for i, j in zip(X, Y):
    mat[i][j] = 1

mat2 = np.rot90(mat, k=-1)
display(mat2)

mat3 = mat[:, ::-1]
display(mat3)
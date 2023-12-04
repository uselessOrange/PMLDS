import numpy as np
loss = {}

datastruct = {'id1':[1,2],'id2':[3,4,5]}

size = np.random.randint(5, 15)
loss = np.random.normal(1, 2, size=size)
datastruct = {'id1':list(loss),'id2':[3,4,5]}
print(datastruct)
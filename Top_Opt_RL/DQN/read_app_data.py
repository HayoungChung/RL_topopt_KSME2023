# Edit by Son
import numpy as np
import json
import matplotlib.pyplot as plt
with open('App_Data.json') as file:
    datas = json.load(file)
    aa = datas["Topology"]

aa_sq = np.array(aa).squeeze().reshape(24,24)
aa_num = aa_sq.astype(np.float64)
aa_num = np.int64(aa_num)

fig, ax = plt.subplots()
ax.matshow(aa_num, cmap=plt.cm.Blues)

for i in range(24):
    for j in range(24):
        c = aa_num[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.savefig('result.jpg', dpi=300)
plt.show()
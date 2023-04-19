'''
Plot the layout of the structure
input: 1. file_name (string, the name of the json file, e.g. 'App_Data.json')
       2. number of elements (tuple, e.g. (24,24))
       3. save_name (string, the name of the saved image, e.g. 'result.jpg')
'''
import numpy as np
import json
import matplotlib.pyplot as plt

def plot_layout(file_name = 'App_Data.json', number_of_elements = (24, 24), save_name = 'result.jpg'):
    with open(file_name) as file:
        datas = json.load(file)
        aa = datas["Topology"]
    
    aa_sq = np.array(aa).squeeze().reshape(number_of_elements[0], number_of_elements[1])
    aa_num = aa_sq.astype(np.float64)
    aa_num = np.int64(aa_num)
    
    fig, ax = plt.subplots()
    ax.matshow(aa_num, cmap=plt.cm.Blues)
    for i in range(number_of_elements[0]):
        for j in range(number_of_elements[1]):
            c = aa_num[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    plt.savefig(save_name, dpi=300)
    plt.show()
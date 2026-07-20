import json
import numpy as np

with open("results.json","r") as f:
    results=json.load(f)

for name, accs in results.items():
    accs=np.array(accs)
    se=accs.std()/np.sqrt(len(accs))
    print(name,"평균:", accs.mean(), "표준편차:",accs.std(),"표준오차:",se)

with open("resultsfashioncnn.json","r") as f:
    results=json.load(f)

for name, accs in results.items():
    accs=np.array(accs)
    se=accs.std()/np.sqrt(len(accs))
    print(name,"평균:", accs.mean(), "표준편차:",accs.std(),"표준오차:",se)

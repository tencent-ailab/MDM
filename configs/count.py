from collections import Counter

import numpy as np

n_nodes = Counter({22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
                   15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
                   8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 1})

print(sum(number * count for number, count in n_nodes.items()))
print(n_nodes.values())
print(np.mean(n_nodes.values()))

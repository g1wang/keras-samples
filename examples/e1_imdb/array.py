import numpy as np
arrs = [
    [1,2,6,4],
    [3,2,1,9]
]
results1 = np.zeros((len(arrs), 10))
results1[0,arrs[0]] = 1
print(results1)
def vectorize_sequence(sequences, dimension=10):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequence(arrs)
print(x_train)

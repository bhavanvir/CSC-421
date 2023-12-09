from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt

def plot_samples(samples, state2colour, title):
    colours = [state2colour[x] for x in samples]
    x = np.arange(0, len(colours))
    y = np.ones(len(colours))
    plt.figure(figsize=(10,1))
    plt.bar(x, y, color=colours, width=1)
    plt.title(title)

np.random.seed(42)

model = hmm.CategoricalHMM(n_components=2)
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.37, 0.63],
                            [0.63, 0.37]])
model.emissionprob_ = np.array([[0.15, 0.35, 0.35, 0.15],
                                [0.40, 0.10, 0.10, 0.40]])

X, Z = model.sample(n_samples=1000)

samples = [item for sublist in X for item in sublist]
states2colour = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}
plot_samples(samples, states2colour, 'ACTG')
obj2colour = {0: 'black', 1: 'white'}
plot_samples(Z, obj2colour, 'CGD and CGS')

plt.show()

X, Z = model.sample(n_samples=10000)

estimated_model = hmm.CategoricalHMM(n_components=2)
estimated_model.fit(X)

print(f"Transition matrix\nEstimated model: {estimated_model.transmat_}\nOriginal model: {model.transmat_}")
print(f"\nEmission probabilities\nEstimated model: {estimated_model.emissionprob_}\nOriginal model: {model.emissionprob_}")
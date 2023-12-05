
# Modify this code as needed 



import numpy as np
from scipy import stats


class Random_Variable: 
    
    def __init__(self, name, values, probability_distribution): 
        self.name = name 
        self.values = values 
        self.probability_distribution = probability_distribution
        if all(type(item) is np.int64 for item in values):
            self.type = 'numeric'
            self.rv = stats.rv_discrete(name = name, values = (values, probability_distribution))
        elif all(type(item) is str for item in values): 
            self.type = 'symbolic'
            self.rv = stats.rv_discrete(name = name, values = (np.arange(len(values)), probability_distribution))
            self.symbolic_values = values 
        else: 
            self.type = 'undefined'
            
    def sample(self,size): 
        if (self.type =='numeric'):
            return self.rv.rvs(size=size)
        elif (self.type == 'symbolic'): 
            numeric_samples = self.rv.rvs(size=size)
            mapped_samples = [self.values[x] for x in numeric_samples]
            return mapped_samples

    def get_name(self):
        return self.name
    
values = np.array([1,2,3,4,5,6])
probabilities_A = np.array([1/6., 1/6., 1/6., 1/6., 1/6., 1/6.])
probabilities_B = np.array([0.0, 0.0, 0/6., 3/6., 3/6., 0/6.])

dieA = Random_Variable('DieA', values, probabilities_A)
dieB = Random_Variable('DieB', values, probabilities_B)


def dice_war(A,B, num_samples = 1000, output=True):
    # your code goes here
    samples_A = A.sample(num_samples)
    samples_B = B.sample(num_samples)

    prob = np.sum(samples_A > samples_B) / num_samples
    res = prob > 0.5

    if output:
        if res:
            print('{} beats {} with probability {}'.format(A.get_name(),
                                                           B.get_name(),
                                                           prob))
        else:
            print('{} beats {} with probability {:.2f}'.format(B.get_name(),
                                                               A.get_name(),
                                                               1.0-prob))
    return (res, prob)


# Add code here to show the non-transitive nature of Red, Green, Blue dice
dice_war(dieA, dieB)

values_RedGreenBlue = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

probabilities_Red = np.array([0.0, 1/3., 0, 1/3., 0.0, 0.0, 0.0, 0.0, 1/3.])
probabilities_Green = np.array([1/3., 0.0, 0.0, 0.0, 0.0, 1/3., 0, 1/3., 0.0])
probabilities_Blue = np.array([0.0, 0.0, 1/3., 0.0, 1/3., 0, 1/3., 0.0, 0.0])

dieRed = Random_Variable('DieRed', values_RedGreenBlue, probabilities_Red)
dieGreen = Random_Variable('DieGreen', values_RedGreenBlue, probabilities_Green)
dieBlue = Random_Variable('DieBlue', values_RedGreenBlue, probabilities_Blue)

dice_war(dieRed, dieGreen)
dice_war(dieGreen, dieBlue)
dice_war(dieBlue, dieRed)
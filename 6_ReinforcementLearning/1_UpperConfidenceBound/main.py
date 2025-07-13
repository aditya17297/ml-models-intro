import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


dataset = pd.read_csv('/Users/adityaagrawal/PycharmProjects/PythonProject/6_ReinforcementLearning/1_UpperConfidenceBound/Ads_CTR_Optimisation.csv')

## UCB

N = 10000 ## number of rounds
d = 10 ## number of adds

ad_selected = []
number_of_selection = [0] * d
sums_of_reward = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if number_of_selection[i] > 0:
            average_award = sums_of_reward[i] / number_of_selection[i]
            delta_i = math.sqrt((3 / 2) * (math.log(n + 1) / number_of_selection[i]))
            upper_bound = average_award + delta_i
        else:
            upper_bound = 1e400

        if max_upper_bound < upper_bound:
            max_upper_bound = upper_bound
            ad = i

    ad_selected.append(ad)
    number_of_selection[ad] += 1
    reward = dataset.values[n][ad]
    sums_of_reward[ad] = sums_of_reward[ad] + reward
    total_reward = total_reward + reward


## visual

plt.hist(ad_selected)
plt.title('Histogram for ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times ad selected')
plt.show()
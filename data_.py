import pandas as pd
import random
import numpy as np
a = pd.read_csv('da.csv')

a.iloc[:]

N_in = 168
gamma_se = 24 * 0 # 0, 24, 48, 72
p_se = 0.15
end = int(N_in - gamma_se)
start = int(end + 1 - N_in * p_se)
scaling_factor = 0.4 # or 2

start

end

a.iloc[start:end, 3]

ramp_rate = 1
# ne = 
# i = 
l_re = 0

a['Scaling_attack'] = a.iloc[:,3]
a['Scaling_attack'] = a.iloc[:,3]

# for i in range(start, end + 1):

a.iloc[start:end, 4] = a.iloc[start:end, 3] * scaling_factor

# for i in range(start, end + 1):
    
#     a.iloc[i, 5] = (1 + l_re * min(abs(i - start), abs(end-i))) * a.iloc[i, 3] 

# a['Scaling_attack'] = a['Total Load'].apply(lambda x : x * scaling_factor)
# a['Ramping_attack'] = a['Total Load'].apply(lambda x : 1 + x * scaling_factor)
# a.iloc[start:end,3] *= scaling_factor

a.to_csv('scaling_attack.csv', index = False)

b = pd.read_csv('scaling_attack.csv')

b['Ramping_attack'] = b.iloc[:,3]
b['RA_labels'] = 0

b

l_re = 1 # 1
b.iloc[:-5 ,5]

b[start-3:end]


for i in range(start, end + 1):
    constt = (1 + l_re * min(abs(i - start), abs(end - i)))
    # print(constt)
    b.iloc[i, 6] = constt * b.iloc[i, 3] 
    b.iloc[i, 7] = 1

l_de = 1.5
p_de = int(b.iloc[:].shape[0] * 0.15)
p_de

b['Random_attack'] = b.iloc[:, 3]
b['RAN_labels'] = 0

b.iloc[:].shape

x = set(random.sample(range(b.iloc[:].shape[0]), p_de))

for i in x:
    b.iloc[i, 8] = l_de * b.iloc[i, 3]
    b.iloc[i, 9] = 1



b.to_csv('data_attacks.csv', index = False)




import numpy as np

a = np.zeros(b.shape[0])
a[start:end] = 1

b['SA_labels'] = a

b.to_csv('scaling_attack.csv', index = False)

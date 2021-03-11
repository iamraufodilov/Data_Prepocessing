#Binarisation
import numpy as np
from sklearn import preprocessing
input_data = np.array(
   [[2.1, -1.9, 5.5],
    [-1.5, 2.4, 3.5],
    [0.5, -7.9, 5.6],
    [5.9, 2.3, -5.8]])
binarized_data = preprocessing.Binarizer(threshold=0.5).transform(input_data)
print("Our binarized data is: ", binarized_data)

#Mean removal
print("Mean: ", input_data.mean(axis=0))
print("Standart Deviation: ", input_data.std(axis=0))
scaled_data = preprocessing.scale(input_data)
print("Data after removing mean", scaled_data.mean(axis=0))
print("Data after removing Standart Deviation", scaled_data.std(axis=0))

#Scaling
data_scaller_minmax = preprocessing.MinMaxScaler()
my_scaled_data = data_scaller_minmax.fit_transform(input_data)
print(my_scaled_data)

#Normalisation

#L1 normalisation
data_normalized_with_l1 = preprocessing.normalize(input_data, norm='l1')
print("Our data normalized with l1 method: ",data_normalized_with_l1)

#L2 normalisation
data_normalized_with_l2 = preprocessing.normalize(input_data, norm='l2')
print("Our data normalized with l2 method: ", data_normalized_with_l2)

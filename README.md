# CNN
The repository contains the python code to implement Convolutional Neural Network using MLP Classifier
The required libraries are imported as follows:

sklearn.datasets.load_breast_cancer(return_X_y=False)

Load and return the breast cancer wisconsin dataset (classification).
The breast cancer dataset is a classic and very easy binary classification dataset.
Classes	2
Samples per class	212(M),357(B)
Samples total	569
Dimensionality	30
Features	real, positive
Parameters:	
return_X_y : boolean, default=False
If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object.

Returns:	
data : Bunch
Dictionary-like object, the interesting attributes are: ‘data’, the data to learn, ‘target’, the classification labels, ‘target_names’, the meaning of the labels, ‘feature_names’, the meaning of the features, and ‘DESCR’, the full description of the dataset.
(data, target) : tuple if return_X_y is True



Record of changes made to original script to make it work with newer versions of Numpy and Scipy.
Notes.
SciPy 1.6.0: simpson was introduced as the new, preferred name. simps was kept around but deprecated.
SciPy 1.14.0: The old simps alias was officially removed from the library.
Numpy 2.0: np.in1d was deprecated and removed. It is necessary to replace in1d with isin.
Numpy 2.0: array.ptp() method is deprecated. Top-level function np.ptp should be used instead.

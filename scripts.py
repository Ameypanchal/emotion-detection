import h5py

# Open the .h5 file in read mode
with h5py.File('model/model_20.weights.h5', 'r') as f:
    # Check the metadata in the file
    print(f.attrs.items())
    attributes = list(f.attrs.items())
    print(attributes)
    for attribute in attributes:
        print(attribute)

from sgd_optimization.random_dataset_generator import generate_random_dataset as grd

dataset, labels, colors = grd(read=True)
print(dataset)
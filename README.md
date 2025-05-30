# Dont run this in your local system (laptop or desktop)

This repo is an example of using machine learning on large dataset.

We have used this model in a dataset with 1168 rows and 73331 columns. 

You can find the dataset in the "Data" directory in the main branch. 

But the dataset uploaded to this repo is reduced version of the original dataset due to 25 mb git limit. 

# The raw data requires some preprocessing.

You can find the preprocessing script in the main branch, "Preprocessing_ASV.py"


After preprocessing you can run the model on the preprocessed darta given by the Preprocessing_ASV.py

The modeling script is also available in the main branch of the repo "ASV_extensive_grid_random_forest_no_normalization.py" 

# The gridsearch is very extensive, more than 20,000 fits (21600 to be precise).

Please run it in your server (EC2, Azure, GCP) or the server your university provides. 

# If you dont have the computing resource use Random Search instead of GridSearch. 

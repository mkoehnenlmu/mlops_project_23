# mlops_project_23

## Project Proposal 
### Overall goal of the project
Development and deployment of a Machine Learning pipeline for the prediction of train delays and deployment to the Google Cloud using continuous integration. The goal is to deploy a lightweight model that saves on cost, i.e. a model where a trade-off of model performance for higher inference speed and lower operational cost is preferable, e.g. by spinning up a delay prediction container on demand.
### Use case
The application should predict the delay of a connection and thereby help users to plan their trips more accurately. Potential users could e.g. check whether their planned trip includes connections with a high risk of delay and ensure that transit times match potential delays.
### Features
Dockerization and deployment in the cloud, Data and model versioning and storage in GCP, multi-objective tuning, continuous integration via GH Actions
Optional features (depending on how much time is left): 
Automatic retraining and deployment when a new version of the data is available
Detection of data drift in a separate scheduled job
Explainability (e.g. features causal for prediction or counterfactual explanations)
### ML Framework
Microsoft LightGBM is a fast, optimized gradient boosting framework. It is a good candidate for this project since it enables us to focus on the development of the ML pipeline while offering many hyperparameters to tune. In order to make up for the arguably more straightforward training, cross validation tuning will be performed to find a well performing yet lightweight model.
Alternatively, we propose to use Pytorch Lightning to train a model for the same purpose.
### Inclusion of LightGBM into the project
LightGBM will be responsible for training and inference tasks in the respective docker containers in the cloud. A big advantage of LightGBM is its speed and small size, making it a good candidate for our goal. 
### Data
As the main data source, the “[dbahn_travels_captures](https://www.kaggle.com/datasets/chemamengibar/dbahn-travels-captures)” dataset from kaggle will be used.
It contains data from all train connections passing 14 different stations in Germany on 16.09.2019. The data includes a train identification, the route and final destination, the planned departure time and platform at the respective station and - if applicable - the delay. The data have been collected by the author from information available on the DB‘s website using web-scraping tools created in Python.
Alternatively, we propose to predict flight delays using the [2019 Airline Delays](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations) dataset.

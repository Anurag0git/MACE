MACE (Major Atmospheric Cherenkov Experiment) Data analysis --> to predict whether captured event is Gamma or Hadron(other than gamma)

* This model uses randomForest classification, whereas MAGICmodel uses neural network, this model is in its initial stages.

* run main.py to create model based on magicdataset.csv
* mace_ml_model.pkl is trained model on magicdataset.csv 
* scaler.pkl is a scaler model
    A scaler is a preprocessing tool used to normalize or standardize feature values so that all input features are on the same scale.
In our case, we used StandardScaler from sklearn.preprocessing to scale the MAGIC dataset's features before training the model.
* app.py contains flask api code

* You are welcome to contribute in the project

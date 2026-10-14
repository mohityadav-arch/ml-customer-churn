# Source Code

`train_model.py` contains the baseline end-to-end training workflow.

It:
- loads the CSV,
- cleans `TotalCharges`,
- separates the target,
- builds preprocessing pipelines,
- trains Logistic Regression, Decision Tree and Random Forest,
- evaluates them using accuracy, precision, recall, F1 and ROC-AUC.

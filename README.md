## XGBoost Breast Cancer Detection with Amazon SageMaker

A machine learning pipeline built on AWS SageMaker using XGBoost to classify breast cancer cases as malignant or benign based on diagnostic features. This project demonstrates end-to-end ML practices including model training, deployment, evaluation, and documentation using the SageMaker Model Registry and Model Cards.

## Project Overview
	•	Algorithm: XGBoost (binary classification)
	•	Dataset: Breast cancer dataset (from SageMaker sample datasets)
	•	Infrastructure: AWS SageMaker built-in XGBoost container (1.7-1)
	•	Features: Real-valued tumor attributes (e.g., radius_mean, texture_mean, etc.)
	•	Goal: Predict breast cancer malignancy based on diagnostic features.

 ## Setup
	1.	Clone the repository:git clone <repo name>.git and then do this-> cd <repo name>
	2.	Set up your SageMaker Studio environment or Jupyter notebook with:
		•	boto3
		•	sagemaker
		•	xgboost
		•	scikit-learn
		•	pandas
	3.	Run the notebook: *.ipynb

### Model Training
	•	Trained using SageMaker built-in XGBoost container.
	•	Hyperparameters used:

{
    "objective": "binary:logistic",
    "max_depth": 5,
    "eta": 0.2,
    "gamma": 4,
    "min_child_weight": 6,
    "subsample": 0.8,
    "num_round": 100
}


	•	Training and validation data stored in S3 under s3://<bucket>/<prefix>/.

### Evaluation Metrics

After extracting the model from model.tar.gz and evaluating on the validation dataset:

Metric	Value
Accuracy	0.96
Precision	0.95
Recall	0.94

These metrics are used in the model card and registry entry.

### Deployment
	•	The trained model is deployed as a real-time SageMaker endpoint.
	•	Endpoint config and endpoint creation are fully automated in the notebook.
	•	Supports text/csv input and output format.

## Model Registry & Card

### Model Registry

The model is versioned and registered under:
	•	Model Package Group: xgboost-breast-cancer-detection
	•	Hosted using: create_model_package, create_model_package_group

### Model Card

A SageMaker Model Card includes:
	•	Model purpose
	•	Owner
	•	Training configuration
	•	Evaluation metrics
	•	Intended use and limitations

"evaluation_details": [
  {
    "name": "Validation Evaluation",
    "evaluation_observation": "Achieved 96% accuracy on validation dataset.",
    "datasets": ["s3://<your-bucket>/validation"],
    "metric_groups": [
      {
        "name": "Binary Classification Metrics",
        "metric_data": [
          {"name": "Accuracy", "type": "number", "value": 0.96},
          {"name": "Precision", "type": "number", "value": 0.95},
          {"name": "Recall", "type": "number", "value": 0.94}
        ]
      }
    ]
  }
]

### Inference Example

from sagemaker.predictor import Predictor

predictor = Predictor(endpoint_name="your-endpoint-name", sagemaker_session=sess)
response = predictor.predict("14.2,20.3,92.5,...")
print(response)

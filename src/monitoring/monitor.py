# Can be part of your CI pipeline or a separate script
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Assume old_data is from a previous DVC version and new_data is the current one
old_data = pd.read_csv('path/to/old/data.csv')
new_data = pd.read_csv('data/raw/dataset.csv')

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(current_data=new_data, reference_data=old_data, column_mapping=None)
data_drift_report.save_html('reports/data_drift/index.html')

if data_drift_report.as_dict()['metrics'][0]['result']['dataset_drift']:
    print("Data drift detected!")
    # Potentially fail a CI build here
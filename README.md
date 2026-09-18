# Stress and Emotion Detection using WESAD Dataset

This project classifies human emotional states (Neutral, Stress, Amusement) using EDA and ECG signals from the WESAD dataset.

## Results
- **Random Forest Accuracy:** 85%
- **SVM Accuracy:** 67%

## How to Run
1. Download the WESAD dataset from Kaggle.
2. Place `.pkl` files in `archive_2/WESAD/Sx/` folder structure.
3. Install requirements: `pip install -r requirements.txt`
4. Run the pipeline: `python main_pipeline.py`
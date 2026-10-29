# Ekurhuleni Safety Hotspot Analytics

A Streamlit frontend for actual Colab model outputs. No sample or fabricated crime observations are included.

## Required files in dashboard_data
- cleaned_crime.csv (Stage 5)
- validation_results.csv (Stage 7)
- model_evaluation.csv (Stage 8)
- confusion_matrix.csv (Stage 8)
- test_predictions.csv (Stage 8)
- station_clusters.csv (Stage 9)

Other exported files can be retained as supporting evidence. Run in project root:

```
pip install -r requirements.txt
streamlit run app.py
```

Model `.pkl` files are optional for this descriptive dashboard, which displays precomputed model outcomes rather than running new predictions. For true live inference, a separate time-safe scoring workflow needs to be added.

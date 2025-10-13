# Forest Cover Type Classification 🌲

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io/)

## Overview

This project implements a multi-class classification pipeline to predict forest cover types (7 classes, e.g., Spruce/Fir, Lodgepole Pine, Aspen) using the **Covertype dataset** from the UCI Machine Learning Repository. It leverages tree-based models to analyze cartographic and environmental features like elevation, slope, soil type, and distances to hydrology/roadways/fire points.

Key goals:
- Clean and preprocess data (outlier capping, categorical consolidation).
- Train/evaluate models: Decision Tree (baseline), Random Forest, XGBoost.
- Visualize confusion matrices and feature importances.
- Bonus: Model comparison and hyperparameter tuning.

Achieved **97.1% accuracy** with XGBoost—enabling applications in environmental monitoring and conservation.

## Dataset

- **Source**: [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype) (581,012 samples, 55 features).
- **Features**: 10 numeric (e.g., Elevation, Slope), 44 binary (soil/wilderness one-hots), 1 target (Cover_Type: 1-7).
- **Download**: Place `covtype.csv` in `./data/` (or load via URL in code).

Class distribution (imbalanced—handled via `class_weight='balanced'`):
- Class 2 (Lodgepole Pine): 283,301
- Class 1 (Spruce/Fir): 211,840
- ... (full in notebook)

## Installation & Dependencies

1. **Clone/Setup**:
   ```
   git clone <your-repo-url>
   cd forest-cover-classification
   ```

2. **Environment** (Python 3.8+):
   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt**:
   ```
   pandas>=1.5.0
   numpy>=1.24.0
   matplotlib>=3.6.0
   seaborn>=0.12.0
   scikit-learn>=1.2.0
   xgboost>=1.7.0
   ```

3. **Data**: Ensure `data/covtype.csv` exists (download from UCI).

## Usage

Run the full pipeline (EDA, preprocessing, modeling, evaluation) via:

```bash
python main.py
```

- **Output**: Prints stats, accuracies, reports; generates plots (histograms, boxplots, CMs, importances).
- **Customization**: Edit `main.py` for params (e.g., `n_estimators=200` in RF/XGB).
- **Jupyter Alternative**: See `notebook.ipynb` for interactive version with inline visuals.

Example run time: ~3-5 min on CPU (581k samples).

## Results

| Model         | Test Accuracy | Macro F1 | Key Strength                  |
|---------------|---------------|----------|-------------------------------|
| Decision Tree | 91.8%        | 0.88    | Fast baseline                |
| Random Forest | 95.7%        | 0.92    | Handles imbalance well       |
| XGBoost       | **97.1%**    | 0.94    | Best overall; robust to noise|

- **Feature Insights**: Elevation (top predictor) and Soil_Type drive classifications—higher elevations favor conifers.
- **Tuning**: RF optimized via RandomizedSearchCV (best: n_estimators=100, max_depth=None; CV acc 94.7%).

Visuals (generated in run):
- Confusion Matrices: Show minor errors between similar classes (e.g., Spruce/Fir vs. Lodgepole).
- Importances: Bars highlight Elevation (~24% in RF), Wilderness Areas (~27% in XGB).

## Key Insights & Conclusion

Tree-based models captured environmental nuances effectively, with XGBoost excelling on imbalanced minorities (e.g., Aspen F1=0.92). Preprocessing (outlier capping, soil consolidation) boosted stability and interpretability.

This pipeline demonstrates supervised learning for ecological classification, supporting conservation (e.g., habitat mapping) and land management.

## License

MIT License—feel free to use/modify.

## Acknowledgments

- Dataset: [Forest Cover Type Dataset on UCI](https://archive.ics.uci.edu/dataset/31/covertype).
- Tools: Scikit-learn, XGBoost, Matplotlib/Seaborn.

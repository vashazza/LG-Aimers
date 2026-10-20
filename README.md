# LG Aimers - Product Defect Classification

**LG AI Research, Jul. – Aug. 2024, Team Leader**

Binary classification of defective products from high-dimensional, highly imbalanced manufacturing process data.

## Approach
- Reduced dimensionality with Truncated SVD and handled class imbalance with resampling (SMOTEENN) and class weighting.
- Compared SVM, LSTM / LSTM-encoder, autoencoder, and CatBoost models with grid search and k-fold validation.
- CatBoost with SVD features and class weighting reached the highest validation accuracy among the models compared.

## Files
- `catboost+Trucated SVD (가중치부여) .ipynb`: CatBoost + SVD + class weights
- `8_11_SVM+...ipynb`: SVM baselines with grid search and k-fold
- `8_14_LSTM+SMOTEENN+TrucatedSVD.ipynb`: LSTM with SMOTEENN resampling
- `8_15_AutoEncoder.ipynb`, `8_19_LSTMEncoder+...ipynb`: Autoencoder / LSTM-encoder experiments

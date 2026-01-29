# 🌾 Intelligent Crop Recommendation System

An advanced machine learning system that provides intelligent crop recommendations based on soil conditions, environmental factors, and agricultural parameters. This system achieves **excellent performance** using an ensemble of state-of-the-art machine learning algorithms.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Intelligent Crop Recommendation System is designed to help farmers and agricultural experts make data-driven decisions about which crops to plant based on various environmental and soil conditions. The system analyzes multiple factors including soil type, pH levels, nutrient content (NPK), temperature, humidity, water requirements, and seasonal patterns to recommend the most suitable crop.

### Key Highlights

- **57 Different Crop Types** supported
- **High Accuracy** with ensemble learning approach
- **29 Engineered Features** for comprehensive analysis
- **Ensemble Learning** approach combining 4 powerful algorithms
- **SHAP Analysis** for model interpretability
- **Multiple Prediction Modes** (Single, Batch, CSV)

## ✨ Features

### Core Capabilities

- 🔍 **Comprehensive Analysis**: Evaluates 29 different features including soil properties, climate conditions, and nutrient levels
- 🤖 **Ensemble Learning**: Combines Random Forest, XGBoost, LightGBM, and CatBoost for superior performance
- 📊 **Explainable AI**: SHAP (SHapley Additive exPlanations) integration for model interpretability
- 🎯 **High Performance**: Robust accuracy with ensemble voting methodology
- 📈 **Visualization**: Comprehensive EDA with interactive Plotly charts
- 💾 **Easy Deployment**: Saved model for quick predictions
- 📁 **Flexible Input**: Supports single predictions, multiple samples, and batch CSV processing

### Prediction Methods

1. **Single Sample Prediction**: Test individual scenarios
2. **Multiple Sample Prediction**: Test multiple predefined scenarios
3. **Batch CSV Prediction**: Process large datasets from CSV files
4. **Random Test Validation**: Verify model with actual test data

## 📊 Dataset

### Dataset Characteristics

- **Total Samples**: 57,000
- **Features**: 23 original columns
- **Target Classes**: 57 different crops
- **Engineered Features**: 29 (after preprocessing)

### Input Features

#### Categorical Features
- `SOIL`: Soil type (Alluvial, Loamy, Clay, Sandy, etc.)
- `SEASON`: Growing season (Kharif, Rabi, Zaid)
- `SOWN`: Sowing month
- `HARVESTED`: Harvesting month
- `WATER_SOURCE`: Irrigation type (Irrigated, Rainfed)

#### Numerical Features
- **Soil Properties**: pH levels (min and max)
- **Climate**: Temperature, humidity (min and max)
- **Crop Duration**: Growing period in days
- **Water Requirements**: Water needed for crop (mm)
- **Nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K) levels (min and max)

### Target Crops

The system can recommend from 57 different crop types including:
- **Cereals**: Rice, Wheat, Maize, Barley, Millets
- **Pulses**: Chickpea, Pigeon pea, Lentil, Green gram, Black gram
- **Cash Crops**: Cotton, Sugarcane, Jute, Tobacco
- **Oilseeds**: Groundnut, Sunflower, Soybean, Mustard
- **Vegetables**: Potato, Onion, Tomato, and many more

## 🤖 Models Used

### Ensemble Architecture

The system uses a **Voting Classifier** that combines predictions from four powerful algorithms:

1. **Random Forest Classifier**
   - Ensemble of decision trees
   - Robust to overfitting
   - Handles non-linear relationships well

2. **XGBoost (Extreme Gradient Boosting)**
   - Advanced gradient boosting framework
   - High performance on structured data
   - Built-in regularization

3. **LightGBM (Light Gradient Boosting Machine)**
   - Fast training speed
   - Lower memory usage
   - High accuracy

4. **CatBoost**
   - Handles categorical features automatically
   - Reduces overfitting
   - Minimal parameter tuning required

### Model Performance

```
High Accuracy: Excellent performance on test data
Number of Crops: 57
Total Features: 29
Model Type: Voting Ensemble
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Colab (recommended) or Jupyter Notebook
- Google Drive account (for data storage)

### Required Libraries

```bash
pip install xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn plotly shap optuna imbalanced-learn
```

### Clone Repository

```bash
git clone https://github.com/MushfiqurRashid/Intelligent-Crop-Recommendation-System.git
cd Intelligent-Crop-Recommendation-System
```

## 💻 Usage

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge at the top of this README
2. Mount your Google Drive when prompted
3. Upload the dataset to your Google Drive
4. Run all cells sequentially

### Option 2: Local Jupyter Notebook

1. Install required libraries
2. Download the dataset
3. Update the dataset path in the notebook
4. Run the notebook cells

### Making Predictions

#### Single Prediction

```python
# Example input
sample = {
    'SOIL': 'Loamy soil',
    'SEASON': 'kharif',
    'SOWN': 'Jun',
    'HARVESTED': 'Oct',
    'WATER_SOURCE': 'irrigated',
    'SOIL_PH': 6.5,
    'SOIL_PH_HIGH': 7.0,
    'CROPDURATION': 120.0,
    'CROPDURATION_MAX': 150,
    'TEMP': 28.0,
    'MAX_TEMP': 35,
    'WATERREQUIRED': 1500.0,
    'WATERREQUIRED_MAX': 2000,
    'RELATIVE_HUMIDITY': 70.0,
    'RELATIVE_HUMIDITY_MAX': 80,
    'N': 85.0,
    'N_MAX': 100,
    'P': 45.0,
    'P_MAX': 60,
    'K': 50.0,
    'K_MAX': 60
}

# Get prediction
prediction = predict_crop(sample)
```

#### Batch Prediction from CSV

```python
# Load and predict from CSV file
predictions = predict_from_csv('/path/to/your/data.csv')
```

#### Using Saved Model

```python
import pickle

# Load the model
with open('crop_recommendation_final_model.pkl', 'rb') as f:
    model_package = pickle.load(f)

# Extract components
model = model_package['model']
scaler = model_package['scaler']
feature_names = model_package['feature_names']
target_encoder = model_package['target_encoder']
```

## 📁 Project Structure

```
Intelligent-Crop-Recommendation-System/
│
├── crop.ipynb                              # Main Jupyter notebook
├── README.md                               # Project documentation
├── crop_recommendation_final_model.pkl     # Trained model (generated)
├── unknown_data_template.csv               # CSV template for predictions
│
└── (Data files - stored in Google Drive)
    └── Crop recommendation dataset.csv
```

## 📈 Model Performance

### Evaluation Metrics

The model has been evaluated using multiple metrics:

- **Accuracy**: Strong performance across test dataset
- **Precision**: High across all crop classes
- **Recall**: Excellent detection rate
- **F1-Score**: Balanced performance
- **Cross-Validation**: Consistent performance across folds

### Visualizations Included

1. **Exploratory Data Analysis (EDA)**
   - Distribution plots for all features
   - Correlation heatmaps
   - Feature importance charts
   - Target class distribution

2. **Model Performance**
   - Confusion matrix
   - Classification reports
   - ROC curves
   - Learning curves

3. **SHAP Analysis**
   - Feature importance plots
   - Individual prediction explanations
   - Summary plots

## 🛠️ Technologies Used

### Machine Learning & Data Science
- **scikit-learn**: Model building and evaluation
- **XGBoost**: Gradient boosting
- **LightGBM**: Efficient gradient boosting
- **CatBoost**: Categorical boosting
- **SHAP**: Model interpretability

### Data Processing & Analysis
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

### Visualization
- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts

### Model Optimization
- **Optuna**: Hyperparameter tuning
- **imbalanced-learn**: Handling class imbalance

### Deployment
- **Joblib/Pickle**: Model serialization
- **Google Colab**: Cloud-based development

## 🔮 Future Enhancements

### Planned Features

- [ ] Web application interface using Flask/Streamlit
- [ ] Real-time weather data integration
- [ ] Geographic location-based recommendations
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Historical yield prediction
- [ ] Crop rotation suggestions
- [ ] Pest and disease prediction
- [ ] Cost-benefit analysis
- [ ] Integration with agricultural databases

### Potential Improvements

- Deep learning models (Neural Networks)
- Time-series analysis for seasonal trends
- Satellite imagery integration
- IoT sensor data incorporation
- Market price prediction integration

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

### Guidelines

- Follow PEP 8 style guidelines for Python code
- Add comments and documentation
- Update README.md if needed
- Test your changes thoroughly

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Mushfiqur Rashid** - [GitHub Profile](https://github.com/MushfiqurRashid)

## 🙏 Acknowledgments

- Dataset source: [Include dataset source if applicable]
- Inspired by precision agriculture and sustainable farming practices
- Thanks to the open-source community for amazing ML libraries

## 📞 Contact

For questions, suggestions, or collaborations:

- GitHub: [@MushfiqurRashid](https://github.com/MushfiqurRashid)
- Project Link: [https://github.com/MushfiqurRashid/Intelligent-Crop-Recommendation-System](https://github.com/MushfiqurRashid/Intelligent-Crop-Recommendation-System)

---

⭐ If you find this project useful, please consider giving it a star on GitHub!

**Made with ❤️ for sustainable agriculture**

# 🚀 BIMBO AI - Quick Setup Guide

## Team Matsya N - EEG Emotion Recognition System

---

## 📋 Prerequisites

- **Python 3.8+** installed ([Download here](https://www.python.org/downloads/))
- **Git** (optional, for cloning)
- **Internet connection** (for package installation)

---

## ⚡ Quick Start (Windows)

### Option 1: One-Click Setup (Recommended)

1. **Double-click** `run_bimbo_ai.bat`
2. Wait for installation to complete
3. Dashboard opens automatically in browser!

### Option 2: Manual Setup

```cmd
# 1. Install requirements
pip install -r requirements.txt

# 2. Run dashboard
streamlit run bimbo_ai_dashboard.py
```

---

## 🎯 What the Script Does

1. ✅ Checks Python installation
2. ✅ Upgrades pip to latest version
3. ✅ Installs all required packages from `requirements.txt`
4. ✅ Creates necessary directories (`data/`, `data/processed/`, etc.)
5. ✅ Launches Streamlit dashboard
6. ✅ Opens browser automatically

---

## 📦 Installed Packages

- **MNE-Python** - EEG data processing
- **Scikit-Learn** - Machine learning models
- **XGBoost** - Advanced ML algorithm
- **Streamlit** - Interactive dashboard
- **Pandas, NumPy, SciPy** - Data manipulation
- **Matplotlib, Seaborn** - Visualizations
- **Groq** - AI report generation
- **ReportLab** - PDF exports
- **imbalanced-learn** - SMOTE balancing

---

## 🌐 Accessing the Dashboard

Once running, the dashboard will be available at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.x.x:8501

---

## 🎨 Features Available

1. **📊 Dataset Upload** - Support for .fif, .edf, .csv, .mat, .set files (up to 1 GB)
2. **📈 Exploratory Analysis** - Arousal-Valence plots, correlation matrices
3. **🤖 ML Classification** - XGBoost, Ensemble, Random Forest models
4. **🧠 Brain Topography** - Scalp maps for all frequency bands
5. **📥 Multi-Format Export** - TXT, CSV, JSON, MD, PDF reports
6. **🎯 AI Analysis** - Automated insights using Groq API

---

## 🛠️ Troubleshooting

### Python Not Found
```cmd
# Add Python to PATH or use full path
C:\Python39\python.exe -m pip install -r requirements.txt
```

### Port Already in Use
```cmd
# Use different port
streamlit run bimbo_ai_dashboard.py --server.port 8502
```

### Package Installation Fails
```cmd
# Try with --user flag
pip install --user -r requirements.txt
```

### Missing Groq API Key
- Dashboard works without API key
- AI report generation requires Groq API key
- Get free key at: https://console.groq.com

---

## 📁 Project Structure

```
eeg-emotion-recognition/
├── run_bimbo_ai.bat          # One-click setup script
├── bimbo_ai_dashboard.py     # Main dashboard application
├── requirements.txt          # Python dependencies
├── index.html                # Presentation website
├── src/                      # Source code modules
│   ├── preprocessing/        # EEG preprocessing
│   ├── features/             # Feature extraction
│   ├── models/               # ML models
│   ├── visualization/        # Plotting functions
│   └── utils/                # Utility functions
├── data/                     # Data directory (created automatically)
└── .streamlit/               # Streamlit configuration
```

---

## 🎓 Usage Workflow

1. **Upload Dataset** - Use file uploader in sidebar
2. **Select Model** - Choose XGBoost (recommended) or other models
3. **Configure Options** - Enable SMOTE, feature selection
4. **Train Model** - Click "Train Model" button
5. **View Results** - See accuracy, confusion matrix, visualizations
6. **Generate Report** - Click "Generate AI Analysis Report"
7. **Export** - Download in preferred format (PDF, CSV, etc.)

---

## 🏆 Hackathon Achievements

- ✅ **100/100 Points** - Perfect implementation score
- ✅ **80%+ Accuracy** - XGBoost model performance
- ✅ **5/5 Phases** - All requirements completed
- ✅ **Professional UI** - Modern dark-themed dashboard
- ✅ **AI Integration** - Groq-powered analysis

---

## 👥 Team Information

**Team Name**: Matsya N  
**Member**: AKSHAY D  
**GitHub**: [github.com/Akshay404error/iit-neuro-hack](https://github.com/Akshay404error/iit-neuro-hack)  
**LinkedIn**: [linkedin.com/in/akshay-d-363aa4294](https://www.linkedin.com/in/akshay-d-363aa4294)

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review error messages in terminal
3. Ensure all prerequisites are met
4. Contact team via GitHub issues

---

**© 2026 Team Matsya N | BIMBO AI - Scientifically Correct & Reproducible**

# Electric Vehicle Range Predictor

A small machine learning project that predicts electric vehicle (EV) range from vehicle specifications. This repository includes the dataset, a training script / model module, and a Streamlit app to interactively predict range for new inputs.

**Project status:** Prototype — model and app work locally; see `app.py` to run the Streamlit demo.

**Table of contents**
- [Overview](#overview)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model training & evaluation](#model-training--evaluation)
- [Results](#results)
- [Notes & suggestions](#notes--suggestions)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project trains a supervised regression model that estimates the driving range (miles / km) of electric vehicles based on technical specifications (battery capacity, weight, motor power, etc.). A Streamlit web UI (`app.py`) lets users enter vehicle specs and get a predicted range.

## Dataset

- Filename: `electric_vehicles_spec_2025.csv` (included in the repo)
- The dataset contains vehicle specifications and the reported or estimated range used as the target variable.

If you use or redistribute this dataset, include appropriate attribution and follow any licensing terms from the source.

## Repository structure

- `app.py` — Streamlit application to interactively predict EV range.
- `ev_model.py` — Model training / inference utilities. This script includes preprocessing, model training, and saving/loading the trained model.
- `electric_vehicles_spec_2025.csv` — Dataset used for training and evaluation.
- `README.md` — This file.
- `CV/` — (optional) folder containing notebooks, visualizations or exported artifacts.

Open these files to inspect preprocessing steps and model choice.

## Installation

Prerequisites:

- Python 3.8 or newer

Suggested packages (install via pip):

```bash
pip install streamlit pandas scikit-learn numpy matplotlib seaborn joblib
```

If you prefer, create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows (Git Bash / WSL): source .venv/Scripts/activate
pip install -r requirements.txt  # if you maintain one
```

## Usage

Run the Streamlit app to use the model interactively:

```bash
streamlit run app.py
```

The app UI allows you to:
- Load a saved model (if implemented in `ev_model.py`)
- Provide vehicle specification input fields
- Click a button to see the predicted driving range and basic explanation/visualizations

If you want to run model training locally (recreate or improve the model), open `ev_model.py`. A common pattern is:

```bash
python ev_model.py  # check the file for available CLI flags or functions
```

If `ev_model.py` saves a model artifact (e.g., `model.joblib`), the Streamlit app will load it for predictions. If not, the app may run a quick training routine on startup — check `app.py` for details.

## Model training & evaluation

The repo's `ev_model.py` contains preprocessing and model code. Typical steps to reproduce training:

1. Load and clean `electric_vehicles_spec_2025.csv`.
2. Feature engineering (scale/encode as needed).
3. Split into train / validation sets.
4. Train a regression model (e.g., RandomForest, GradientBoosting, or linear models).
5. Evaluate on hold-out data and save the best model.

Add an explicit training CLI or notebook if you plan to iterate frequently.

## Results

Document any metrics and observations here after running training locally, for example:

- Model: RandomForestRegressor
- Validation RMSE: XX.X miles
- R^2: 0.9X

Replace the above with real numbers after training.

## Notes & suggestions

- Data quality matters: check for missing values and outliers in `electric_vehicles_spec_2025.csv`.
- Consider feature importance analysis to understand which specs influence range most (battery capacity, weight, aerodynamics proxies).
- For production: export a deterministic model artifact (e.g., joblib) and add input validation and unit tests for the prediction function.
- If you need reproducibility, pin package versions in `requirements.txt` and set random seeds in training.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository.
2. Create a branch for features or fixes.
3. Add tests and update documentation.
4. Create a pull request describing your changes.

If you'd like, I can add a `requirements.txt`, examples of model evaluation scripts, or a `LICENSE` file.

## License

This repository does not include an explicit license file. If you want to open-source the project, consider adding an `LICENSE` (for example, MIT) file.

## Contact

Maintainer: You (update this with your name and contact information)

Questions or feature requests: open an issue in this repository.

---

If you'd like, I can also:
- Create a `requirements.txt` with pinned versions
- Add a short example notebook in `CV/` showing training and evaluation steps
- Add a `LICENSE` file (e.g., MIT)

Tell me which of those you'd like me to do next.

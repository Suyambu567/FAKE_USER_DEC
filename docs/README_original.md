# Fake Profile Detection Project

This project trains a machine learning model to classify Instagram account types (fake vs real) based on profile features, and provides a Flask web interface for making predictions.

## Project Structure

- `FAKE_PROFILE_TRAIN_CODE/` – Contains the training script (`train.py`) and utilities.
- `website/` – Contains the Flask application (`app.py`) and front‑end templates.
- `dataset.csv` – The dataset used for training (located in `website/`).
- Model artifacts (`trained_model.pkl`, `bio_encoder.pkl`, `account_encoder.pkl`) are generated during training and used by the Flask app.

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone or download this repository.
2. Install the required Python packages:

   ```bash
   pip install -r website/requirements.txt
   ```

   *Note:* The training script may require additional packages (`pandas`, `scikit-learn`, `joblib`). They are included in the website requirements.

## Training the Model

The model is trained using the script in `FAKE_PROFILE_TRAIN_CODE/train.py`.

1. Ensure the dataset is accessible. The script expects `dataset.csv` in the same directory as `train.py`. You can copy the dataset from the website folder:

   ```bash
   cp website/dataset.csv FAKE_PROFILE_TRAIN_CODE/
   ```

2. Change to the training directory and run the script:

   ```bash
   cd FAKE_PROFILE_TRAIN_CODE
   python train.py
   ```

   The script will:
   - Load and preprocess the data.
   - Train a Random Forest classifier.
   - Save the trained model and encoders as:
     - `trained_model.pkl`
     - `bio_encoder.pkl`
     - `account_encoder.pkl`

3. (Optional) Verify that the three `.pkl` files appear in `FAKE_PROFILE_TRAIN_CODE/`.

## Deploying the Flask Web App

The Flask app (`website/app.py`) loads the model and encoders from its own directory. After training, copy the generated model files into the `website/` folder:

```bash
copy FAKE_PROFILE_TRAIN_CODE\trained_model.pkl website\
copy FAKE_PROFILE_TRAIN_CODE\bio_encoder.pkl website\
copy FAKE_PROFILE_TRAIN_CODE\account_encoder.pkl website\
```

*(On Unix‑based systems use `cp` instead of `copy`.)*

### Running the App

1. Navigate to the website directory:

   ```bash
   cd website
   ```

2. Start the Flask development server:

   ```bash
   python app.py
   ```

   By default the app runs on **http://127.0.0.1:5000/** (port 5000).

### Changing the Port

To run the app on a different port, modify the `app.run()` call in `website/app.py`:

```python
if __name__ == '__main__':
    app.run(debug=True, port=8080)   # Example: port 8080
```

Alternatively, set the `FLASK_RUN_PORT` environment variable before starting:

```bash
set FLASK_RUN_PORT=8080   # Windows
export FLASK_RUN_PORT=8080 # Linux/macOS
python app.py
```

## Usage

1. Open a web browser and go to the URL shown in the console (e.g., http://127.0.0.1:5000/).
2. Fill in the profile features in the form:
   - Followers, Following, Posts, Engagement Rate (%),
   - Avg Likes per Post, Avg Comments per Post,
   - Verified (0 or 1), Account Age (Years),
   - Bio Text.
3. Submit the form to get a prediction (Fake or Real) along with a confidence score.

## Notes

- The model files must be present in the same directory as `app.py`; otherwise the app will show an error.
- For production use, consider disabling `debug=True` and using a production WSGI server (e.g., Gunicorn).

## Troubleshooting

- **Model not found**: Ensure the three `.pkl` files are in the `website/` folder and that the filenames match exactly.
- **Port already in use**: Choose a different port or stop the existing service.
- **Dependency issues**: Re‑install packages with `pip install -r website/requirements.txt`.

--- 

Enjoy detecting fake profiles!


## how to run the folder

1️⃣   Prerequisites

  ┌───────────────────────────────────┬──────────────────────────────────────────────────────┐
  │           What you need           │                         Why                          │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Python 3.8+                       │ The project uses recent Python syntax and libraries. │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Git (optional)                    │ To clone/download the repo if you haven’t already.   │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Internet access (first time only) │ To download the required Python packages.            │
  └───────────────────────────────────┴──────────────────────────────────────────────────────┘

  ▎ Tip: The repository already contains a virtual‑environment folder (.venv). If you prefer to use it, skip
  ▎ the “Create a new venv” step and just activate the existing one.

  ---
  2️⃣   Clone / locate the project

  If you downloaded the ZIP, extract it and open the folder in your terminal/command prompt.
  If you prefer Git:

  git clone <repository‑url>   # replace with the actual repo URL
  cd FAKE_USER_DEC             # the folder you just cloned/extracted

  You should see the following top‑level items:

  FAKE_USER_DEC/
  │
  ├── FAKE_PROFILE_TRAIN_CODE/   # training scripts & dataset
  ├── website/                   # Flask app + templates
  ├── requirements.txt           # Python dependencies (in website/)
  ├── README.md                  # project overview
  └── ... (other files)

  ---
  3️⃣   Set up the Python environment

  Option A – Use the existing virtual environment (recommended)

  # From the project root:
  .\.venv\Scripts\activate   # Windows PowerShell/CMD
  # or, if you use Git Bash:
  source .venv/Scripts/activate

  You should see (.venv) prefixed to your prompt.

  Option B – Create a fresh venv (if you deleted .venv)

  python -m venv .venv          # create a new virtual environment
  .\.venv\Scripts\activate      # activate it (Windows)
  # source .venv/bin/activate   # macOS / Linux

  ---
  4️⃣   Install dependencies

  All required packages are listed in website/requirements.txt.

  pip install -r website/requirements.txt

  This installs Flask, pandas, scikit‑learn, joblib, etc.

  ▎ Note: If you see any permission errors, run the command prompt as Administrator or add --user to the pip
  ▎ command.

  ---
  5️⃣   (Optional) Train / verify the model

  The repository already ships the trained model files inside the website/ folder:

  - trained_model.pkl
  - account_encoder.pkl
  - bio_encoder.pkl

  If those files are missing or you want to retrain with a different dataset, run the training script:

  cd FAKE_PROFILE_TRAIN_CODE
  python train.py          # or python train_improved.py for the “improved” version

  After training finishes, copy the three .pkl files from FAKE_PROFILE_TRAIN_CODE/ into website/ (overwrite if
  they exist):

  copy trained_model.pkl       ..\website\
  copy account_encoder.pkl     ..\website\
  copy bio_encoder.pkl         ..\website\

  Now return to the project root:

  cd ..

  ---
  6️⃣   Start the Flask web application

  cd website
  python app.py

  You should see output similar to:

  Starting Fake Account Detector web application...
  Access the application at: http://127.0.0.1:5000

  The app runs in debug mode (auto‑reload on code changes).
  If you want to run it on a different port, either:

  - Edit line 278‑280 in website/app.py and change port=5000 to your desired number, or
  - Set an environment variable before starting:

  set FLASK_RUN_PORT=8080   # Windows CMD
  $env:FLASK_RUN_PORT=8080  # PowerShell
  export FLASK_RUN_PORT=8080 # Linux/macOS
  python app.py

  ---
  7️⃣   Open the application

  Open your web browser and go to:

  http://127.0.0.1:5000

  You’ll see the home page with the sidebar navigation. From there you can:

  - Predict – fill in a profile’s features and click Analyze Account.
  - Dashboard – view summary statistics.
  - Analytics – see model performance charts.
  - Settings / Profile / Word Analysis – explore the other pages.

  ---
  8️⃣   Stopping the server

  When you’re done, press Ctrl + C in the terminal where python app.py is running.

  ---
  📌 Quick checklist

  ┌─────┬────────────────────────────────────────────────────────────┐
  │ ✅  │                            Step                            │
  ├─────┼────────────────────────────────────────────────────────────┤
  │ 1   │ Activate the virtual environment (.venv\Scripts\activate). │
  ├─────┼────────────────────────────────────────────────────────────┤
  │ 2   │ pip install -r website/requirements.txt                    │
  ├─────┼────────────────────────────────────────────────────────────┤
  │ 3   │ Verify model files exist in website/ (or train them).      │
  ├─────┼────────────────────────────────────────────────────────────┤
  │ 4   │ cd website && python app.py                                │
  ├─────┼────────────────────────────────────────────────────────────┤
  │ 5   │ Open http://127.0.0.1:5000 in a browser.                   │
  └─────┴────────────────────────────────────────────────────────────┘

  ---
  Troubleshooting tips

  ┌────────────────────────────┬─────────────────────────────┬───────────────────────────────────────────┐
  │          Symptom           │        Likely cause         │                    Fix                    │
  ├────────────────────────────┼─────────────────────────────┼───────────────────────────────────────────┤
  │ ModuleNotFoundError: No    │ Dependencies not installed  │ Ensure the venv is activated and run pip  │
  │ module named 'flask'       │ or wrong env                │ install -r website/requirements.txt.      │
  ├────────────────────────────┼─────────────────────────────┼───────────────────────────────────────────┤
  │ FileNotFoundError: Model   │ .pkl files missing from     │ Run the training script (train.py) and    │
  │ files not found            │ website/                    │ copy the three .pkl files into website/.  │
  ├────────────────────────────┼─────────────────────────────┼───────────────────────────────────────────┤
  │ Page loads but shows       │ Model files present but     │ Re‑train the model with the same versions │
  │ “Model not available”      │ unable to load (e.g.,       │  of scikit‑learn/joblib as in             │
  │                            │ pickle version mismatch)    │ requirements.txt.                         │
  ├────────────────────────────┼─────────────────────────────┼───────────────────────────────────────────┤
  │ Port already in use        │ Another process is          │ Change the port via FLASK_RUN_PORT or     │
  │                            │ listening on 5000           │ stop the conflicting process.             │
  ├────────────────────────────┼─────────────────────────────┼───────────────────────────────────────────┤
  │ Styling looks broken       │ Browser cached old CSS      │ Hard‑refresh (Ctrl + F5) or clear cache.  │
  └────────────────────────────┴─────────────────────────────┴───────────────────────────────────────────┘

  ---
  That’s it! You now have a fully functional Fake Profile Detection web app running locally. Enjoy exploring
  the UI and trying out predictions on different Instagram‑style profile data. 🚀
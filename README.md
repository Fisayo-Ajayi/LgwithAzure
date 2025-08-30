# SuccessLG Heroku Flask ML API

This repository contains a Flask API for machine learning models (KMeans and Scaler), ready to deploy on **Heroku**.

## Project Structure
SuccessLG_Heroku/
│── index.py
│── requirements.txt
│── runtime.txt
│── Procfile
│── models/
│ ├── kmeans.pkl
│ └── scaler.pkl

markdown
Copy code

## Deployment Steps (Heroku)

1. **Create Heroku App**
   - Go to [Heroku Dashboard](https://dashboard.heroku.com/) → New → Create App  
   - Choose a unique name (e.g., `successlg-api`)  
   - Region → closest to you  

2. **Connect GitHub Repo**
   - App → Deploy → Deployment method → GitHub  
   - Select your repo + branch → Enable Automatic Deploys  

3. **Manual Deploy (Optional)**
   - Deploy Branch in the Deploy tab  

4. **Set Environment**
   - Heroku auto-detects `runtime.txt` for Python 3.11  
   - `Procfile` tells Heroku to run Gunicorn:  
     ```
     web: gunicorn index:app --log-file -
     ```

5. **Test Endpoints**
   - Home route:  
     ```
     GET https://<your-app-name>.herokuapp.com/
     ```
     Response:  
     ```json
     {"message": "✅ SuccessLG API running on Heroku"}
     ```

   - Predict route:  
     ```
     POST https://<your-app-name>.herokuapp.com/predict/kmeans
     POST https://<your-app-name>.herokuapp.com/predict/scaler
     ```
     Example JSON body:
     ```json
     {
       "features": [45, 60000, 70, 0.85]
     }
     ```

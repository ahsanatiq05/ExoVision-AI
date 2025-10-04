import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
app.secret_key = "exoplanet_secret_key"

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Load the trained models
try:
    with open("xgb_exoplanet_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("decision_tree_model.pkl", "rb") as f:
        dt_model = pickle.load(f)
    with open("decision_tree_pipeline.pkl", "rb") as f:
        dt_pipeline = pickle.load(f)
    models_loaded = True
except:
    models_loaded = False
    print("Warning: Model files not found. Please ensure model files are in the same directory.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    """Preprocess the input data in the same way as the training pipeline"""
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Convert categorical columns to uppercase and strip whitespace
    for col in df_processed.select_dtypes(include="object").columns:
        df_processed[col] = df_processed[col].str.strip().str.upper()
    
    # Define the columns we need for prediction
    required_columns = [
        'sy_snum', 'sy_pnum', 'disc_year', 'pl_orbper', 'pl_orbsmax', 
        'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj', 'pl_orbeccen', 
        'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 
        'st_logg', 'sy_dist'
    ]
    
    # Check if all required columns are present
    missing_cols = [col for col in required_columns if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Handle missing values - fill with median for numeric columns
    num_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Feature engineering
    df_processed["star_planet_size"] = df_processed["st_rad"] * df_processed["pl_rade"]
    df_processed["temp_diff"] = df_processed["st_teff"] - df_processed["pl_eqt"]
    df_processed["log_insol"] = np.log10(df_processed["pl_insol"].replace(0, np.nan))
    df_processed["planets_per_system"] = df_processed["sy_pnum"]
    df_processed["stars_in_system"] = df_processed["sy_snum"]
    
    # One-hot encode discoverymethod if present
    if 'discoverymethod' in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=["discoverymethod"], drop_first=True)
        
        # Ensure all discoverymethod columns are present
        if 'discoverymethod_MICROLENSING' not in df_processed.columns:
            df_processed['discoverymethod_MICROLENSING'] = 0
        if 'discoverymethod_RADIAL VELOCITY' not in df_processed.columns:
            df_processed['discoverymethod_RADIAL VELOCITY'] = 0
        if 'discoverymethod_TRANSIT' not in df_processed.columns:
            df_processed['discoverymethod_TRANSIT'] = 0
    else:
        # Add default discoverymethod columns if not present
        df_processed['discoverymethod_MICROLENSING'] = 0
        df_processed['discoverymethod_RADIAL VELOCITY'] = 0
        df_processed['discoverymethod_TRANSIT'] = 1  # Default to TRANSIT
    
    # Select only the columns needed for prediction
    prediction_columns = [
        'sy_snum', 'sy_pnum', 'disc_year', 'pl_orbper', 'pl_orbsmax', 
        'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj', 'pl_orbeccen', 
        'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 
        'st_logg', 'sy_dist', 'star_planet_size', 'temp_diff', 'log_insol',
        'planets_per_system', 'stars_in_system', 'discoverymethod_MICROLENSING',
        'discoverymethod_RADIAL VELOCITY', 'discoverymethod_TRANSIT'
    ]
    
    # Ensure all columns are present and in the right order
    for col in prediction_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    df_final = df_processed[prediction_columns]
    
    # Handle any remaining NaN values
    imputer = SimpleImputer(strategy="median")
    df_final = pd.DataFrame(imputer.fit_transform(df_final), columns=df_final.columns)
    
    return df_final

def create_visualizations(df, predictions=None, prediction_probs=None):
    """Create various visualizations for the data"""
    visualizations = {}
    
    # 1. Distribution of key features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distribution of Key Features', fontsize=16)
    
    # Orbital period
    axes[0, 0].hist(df['pl_orbper'], bins=20, alpha=0.7)
    axes[0, 0].set_title('Orbital Period (days)')
    axes[0, 0].set_xlabel('Days')
    axes[0, 0].set_ylabel('Frequency')
    
    # Planet radius
    axes[0, 1].hist(df['pl_rade'], bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Planet Radius (Earth radii)')
    axes[0, 1].set_xlabel('Earth Radii')
    axes[0, 1].set_ylabel('Frequency')
    
    # Stellar temperature
    axes[1, 0].hist(df['st_teff'], bins=20, alpha=0.7, color='orange')
    axes[1, 0].set_title('Stellar Temperature (K)')
    axes[1, 0].set_xlabel('Kelvin')
    axes[1, 0].set_ylabel('Frequency')
    
    # System distance
    axes[1, 1].hist(df['sy_dist'], bins=20, alpha=0.7, color='red')
    axes[1, 1].set_title('System Distance (parsecs)')
    axes[1, 1].set_xlabel('Parsecs')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    visualizations['distributions'] = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    
    # 2. Correlation heatmap
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
    plt.title('Correlation Heatmap of Features', fontsize=16)
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    visualizations['correlation'] = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    
    # 3. If predictions are available, create prediction visualizations
    if predictions is not None:
        # Prediction distribution
        plt.figure(figsize=(8, 6))
        prediction_counts = pd.Series(predictions).value_counts()
        plt.bar(['Candidate', 'Confirmed'], prediction_counts.values, color=['red', 'green'])
        plt.title('Prediction Distribution', fontsize=16)
        plt.ylabel('Count')
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        visualizations['prediction_distribution'] = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()
        
        # If probabilities are available, create ROC curve
        if prediction_probs is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(predictions, prediction_probs)
            plt.plot(fpr, tpr, linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([0, 1, 0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve', fontsize=16)
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            visualizations['roc_curve'] = base64.b64encode(img.getvalue()).decode('utf-8')
            plt.close()
    
    return visualizations

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    return metrics

@app.route('/')
def index():
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Read the CSV file
                df = pd.read_csv(filepath)
                
                # Check if the file has the required columns
                required_columns = [
                    'sy_snum', 'sy_pnum', 'disc_year', 'pl_orbper', 'pl_orbsmax', 
                    'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj', 'pl_orbeccen', 
                    'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 
                    'st_logg', 'sy_dist'
                ]
                
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    flash(f'CSV file is missing required columns: {", ".join(missing_cols)}')
                    return redirect(request.url)
                
                # Preprocess the data
                df_processed = preprocess_data(df)
                
                # Make predictions
                xgb_predictions = xgb_model.predict(df_processed)
                xgb_probabilities = xgb_model.predict_proba(df_processed)[:, 1]
                
                dt_predictions = dt_model.predict(df_processed)
                dt_probabilities = dt_model.predict_proba(df_processed)[:, 1]
                
                # Create visualizations
                visualizations = create_visualizations(df, xgb_predictions, xgb_probabilities)
                
                # Calculate metrics if true labels are available
                metrics = None
                if 'disposition' in df.columns:
                    y_true = df['disposition'].apply(lambda x: 1 if x.upper() == "CONFIRMED" else 0)
                    xgb_metrics = calculate_metrics(y_true, xgb_predictions, xgb_probabilities)
                    dt_metrics = calculate_metrics(y_true, dt_predictions, dt_probabilities)
                    metrics = {
                        'xgb': xgb_metrics,
                        'dt': dt_metrics
                    }
                
                # Prepare data for display
                df_display = df.copy()
                df_display['xgb_prediction'] = ['Confirmed' if p == 1 else 'Candidate' for p in xgb_predictions]
                df_display['xgb_confidence'] = [f"{p:.2%}" for p in xgb_probabilities]
                df_display['dt_prediction'] = ['Confirmed' if p == 1 else 'Candidate' for p in dt_predictions]
                df_display['dt_confidence'] = [f"{p:.2%}" for p in dt_probabilities]
                
                # Convert dataframe to HTML for display
                df_html = df_display.head(100).to_html(classes='table table-striped table-hover', index=False)
                
                return render_template('results.html', 
                                      df_html=df_html,
                                      visualizations=visualizations,
                                      metrics=metrics,
                                      filename=filename,
                                      shape=df.shape)
            
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/manual_input', methods=['GET', 'POST'])
def manual_input():
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'sy_snum': float(request.form['sy_snum']),
                'sy_pnum': float(request.form['sy_pnum']),
                'disc_year': int(request.form['disc_year']),
                'pl_orbper': float(request.form['pl_orbper']),
                'pl_orbsmax': float(request.form['pl_orbsmax']),
                'pl_rade': float(request.form['pl_rade']),
                'pl_radj': float(request.form['pl_radj']),
                'pl_bmasse': float(request.form['pl_bmasse']),
                'pl_bmassj': float(request.form['pl_bmassj']),
                'pl_orbeccen': float(request.form['pl_orbeccen']),
                'pl_insol': float(request.form['pl_insol']),
                'pl_eqt': float(request.form['pl_eqt']),
                'st_teff': float(request.form['st_teff']),
                'st_rad': float(request.form['st_rad']),
                'st_mass': float(request.form['st_mass']),
                'st_met': float(request.form['st_met']),
                'st_logg': float(request.form['st_logg']),
                'sy_dist': float(request.form['sy_dist']),
                'discoverymethod': request.form['discoverymethod'].upper()
            }
            
            # Create dataframe from form data
            df = pd.DataFrame([data])
            
            # Preprocess the data
            df_processed = preprocess_data(df)
            
            # Make predictions
            xgb_prediction = xgb_model.predict(df_processed)[0]
            xgb_probability = xgb_model.predict_proba(df_processed)[0, 1]
            
            dt_prediction = dt_model.predict(df_processed)[0]
            dt_probability = dt_model.predict_proba(df_processed)[0, 1]
            
            # Prepare results
            results = {
                'xgb_prediction': 'Confirmed' if xgb_prediction == 1 else 'Candidate',
                'xgb_confidence': f"{xgb_probability:.2%}",
                'dt_prediction': 'Confirmed' if dt_prediction == 1 else 'Candidate',
                'dt_confidence': f"{dt_probability:.2%}",
                'data': data
            }
            
            return render_template('manual_results.html', results=results)
        
        except Exception as e:
            flash(f'Error processing input: {str(e)}')
            return redirect(request.url)
    
    return render_template('manual_input.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'hirefair-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def compute_metrics(df_test, predictions, y_test):
    results = {}
    groups = df_test['group'].unique()
    group_stats = []
    
    for g in groups:
        mask = df_test['group'] == g
        if mask.sum() == 0:
            continue
        group_pred = predictions[mask]
        group_true = y_test[mask]
        selection_rate = group_pred.mean()
        tn, fp, fn, tp = confusion_matrix(group_true, group_pred, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        group_stats.append({
            'group': g,
            'count': mask.sum(),
            'selection_rate': selection_rate,
            'tpr': tpr,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    if not group_stats:
        return results
    
    selection_rates = [gs['selection_rate'] for gs in group_stats]
    disparate_impact = min(selection_rates) / max(selection_rates) if max(selection_rates) > 0 else 0
    
    tprs = [gs['tpr'] for gs in group_stats]
    equal_opportunity_gap = max(tprs) - min(tprs) if len(tprs) > 1 else 0
    
    results['group_stats'] = group_stats
    results['disparate_impact'] = disparate_impact
    results['equal_opportunity_gap'] = equal_opportunity_gap
    results['disparate_impact_flag'] = disparate_impact >= 0.8
    results['accuracy'] = accuracy_score(y_test, predictions)
    
    return results

def debias_term_scrubbing(text):
    gendered_words = ['he', 'him', 'his', 'she', 'her', 'hers', 'mr', 'mrs', 'ms', 'miss']
    names = ['john', 'michael', 'james', 'robert', 'mary', 'patricia', 'jennifer', 'linda']
    pronouns_patterns = []
    # Simple scrubbing
    text_lower = text.lower()
    for w in gendered_words + names:
        text_lower = text_lower.replace(' ' + w + ' ', ' [redacted] ')
        text_lower = text_lower.replace(w + ' ', '[redacted] ')
        text_lower = text_lower.replace(' ' + w, ' [redacted]')
    return text_lower

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audit', methods=['POST'])
def audit():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    df = pd.read_csv(file)
    session['df'] = df.to_json()
    
    X = df['resume_text']
    y = df['label']
    groups = df['group']
    
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, stratify=y
    )
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    predictions = model.predict(X_test_vec)
    
    test_df = pd.DataFrame({'group': groups_test.values})
    metrics = compute_metrics(test_df, predictions, y_test.values)
    
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_pos_idx = np.argsort(coefs)[-10:][::-1]
    top_pos_words = [(feature_names[i], coefs[i]) for i in top_pos_idx]
    
    session['vectorizer'] = json.dumps({'vocab': vectorizer.vocabulary__, 'idf': vectorizer.idf_.tolist()})
    session['baseline_metrics'] = json.dumps(metrics)
    session['baseline_top_words'] = json.dumps([(w, float(c)) for w,c in top_pos_words])
    session['baseline_accuracy'] = float(metrics['accuracy'])
    
    return render_template('audit.html', metrics=metrics, top_words=top_pos_words, accuracy=metrics['accuracy'])

@app.route('/debias', methods=['POST'])
def debias():
    method = request.form.get('method', 'term_scrubbing')
    df_json = session.get('df')
    if not df_json:
        return redirect(url_for('index'))
    
    df = pd.read_json(df_json)
    X = df['resume_text']
    y = df['label']
    groups = df['group']
    
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, stratify=y
    )
    
    if method == 'term_scrubbing':
        X_train_debiased = X_train.apply(debias_term_scrubbing)
    elif method == 'reweighting':
        # Compute weights for underrepresented group-label combinations
        weights = np.ones(len(X_train))
        group_label_counts = df.groupby(['group', 'label']).size()
        total = len(df)
        for idx, (g, l) in enumerate(zip(groups_train, y_train)):
            count = group_label_counts.get((g, l), 1)
            weights[idx] = total / (len(group_label_counts) * count)
        X_train_debiased = X_train
        sample_weight = weights
    elif method == 'counterfactual_augmentation':
        X_train_list = []
        y_train_list = []
        groups_train_list = []
        for text, label, group in zip(X_train, y_train, groups_train):
            X_train_list.append(text)
            y_train_list.append(label)
            groups_train_list.append(group)
            # Swap he/she
            text_swap = text.replace(' he ', ' she ').replace('He ', 'She ').replace(' him ', ' her ').replace(' his ', ' her ')
            X_train_list.append(text_swap)
            y_train_list.append(label)
            groups_train_list.append(group)
        X_train_debiased = pd.Series(X_train_list)
        y_train_debiased = pd.Series(y_train_list)
        groups_train_debiased = pd.Series(groups_train_list)
    else:
        X_train_debiased = X_train
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    if method == 'reweighting':
        X_train_vec = vectorizer.fit_transform(X_train_debiased)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train, sample_weight=sample_weight)
    elif method == 'counterfactual_augmentation':
        X_train_vec = vectorizer.fit_transform(X_train_debiased)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train_debiased)
    else:
        X_train_vec = vectorizer.fit_transform(X_train_debiased)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
    
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    
    test_df = pd.DataFrame({'group': groups_test.values})
    metrics = compute_metrics(test_df, predictions, y_test.values)
    
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_pos_idx = np.argsort(coefs)[-10:][::-1]
    top_pos_words = [(feature_names[i], coefs[i]) for i in top_pos_idx]
    
    baseline_metrics = json.loads(session.get('baseline_metrics', '{}'))
    baseline_accuracy = session.get('baseline_accuracy', 0)
    baseline_top_words = json.loads(session.get('baseline_top_words', '[]'))
    
    return render_template('compare.html', 
                         method=method,
                         baseline_metrics=baseline_metrics,
                         debiased_metrics=metrics,
                         baseline_accuracy=baseline_accuracy,
                         debiased_accuracy=metrics['accuracy'],
                         baseline_top_words=baseline_top_words,
                         debiased_top_words=[(w, float(c)) for w,c in top_pos_words])

@app.route('/report')
def report():
    baseline_metrics = json.loads(session.get('baseline_metrics', '{}'))
    return render_template('report.html', metrics=baseline_metrics)

if __name__ == '__main__':
    app.run(debug=True)

import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Optional heavy deps
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# Resume parsing
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from docx import Document
except Exception:
    Document = None

# ---- Global UI polish ----
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem;
        border-radius: 8px 8px 0 0;
        background: #f0f2f6;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {background: #1f77b4; color: white;}
    .file-card {border:1px solid #e1e4e8;border-radius:8px;padding:1rem;margin-bottom:0.8rem;background:#fafbfc;}
    .file-card header {display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;}
    .file-name {font-weight:600;color:#24292e;}
    .file-meta {font-size:0.85rem;color:#586069;}
    .preview-box {max-height:180px;overflow:auto;background:#fff;border:1px solid #e1e4e8;border-radius:6px;padding:0.75rem;font-family:monospace;font-size:0.85rem;white-space:pre-wrap;}
    .stButton>button {border-radius:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_text(file):
    name = file.name.lower()
    if name.endswith('.pdf') and pdfplumber:
        with pdfplumber.open(file) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    if name.endswith('.docx') and Document:
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    # txt/csv fallback
    return file.read().decode('utf-8', errors='ignore')


def safe_split(X, y, groups, test_size=0.2, random_state=42):
    """Try stratified split; fall back to random if a class has <2 samples."""
    try:
        return train_test_split(
            X, y, groups, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        return train_test_split(
            X, y, groups, test_size=test_size, random_state=random_state, stratify=None
        )


# ---- Core ML functions ----
def compute_metrics(groups, y_true, y_pred):
    results = {}
    group_stats = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() == 0:
            continue
        gp = y_pred[mask]
        gt = y_true[mask]
        sel = gp.mean()
        tn, fp, fn, tp = confusion_matrix(gt, gp, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        group_stats.append({
            'group': g, 'count': int(mask.sum()),
            'selection_rate': float(sel), 'tpr': float(tpr)
        })
    if not group_stats:
        return {}
    sels = [gs['selection_rate'] for gs in group_stats]
    tprs = [gs['tpr'] for gs in group_stats]
    results['group_stats'] = group_stats
    results['disparate_impact'] = float(min(sels) / max(sels)) if max(sels) > 0 else 0.0
    results['equal_opportunity_gap'] = float(max(tprs) - min(tprs)) if len(tprs) > 1 else 0.0
    results['accuracy'] = float(accuracy_score(y_true, y_pred))
    return results

def debias_term_scrubbing(text):
    gendered = ['he','him','his','she','her','hers','mr','mrs','ms','miss']
    names = ['john','michael','james','robert','mary','patricia','jennifer','linda',
             'sarah','emily','lisa','david','robert','william']
    t = text.lower()
    for w in gendered + names:
        t = t.replace(f' {w} ', ' [redacted] ')
        t = t.replace(f'{w} ', '[redacted] ')
        t = t.replace(f' {w}', ' [redacted]')
    return t

# TF-IDF + LogisticRegression
def train_tfidf_lr(X_train, y_train, vectorizer=None, sample_weight=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
    else:
        X_train_vec = vectorizer.transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train, sample_weight=sample_weight)
    return model, vectorizer

def get_top_words_tfidf(model, vectorizer, n=10):
    feats = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    idx = np.argsort(coefs)[-n:][::-1]
    return [(feats[i], float(coefs[i])) for i in idx]

# Embedding + XGBoost
_encoder = None
def get_encoder():
    global _encoder
    if _encoder is None and SentenceTransformer is not None:
        _encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _encoder

def train_embed_xgb(X_train_texts, y_train, sample_weight=None):
    enc = get_encoder()
    if enc is None:
        raise RuntimeError("sentence-transformers not installed")
    X_train_emb = enc.encode(X_train_texts.tolist(), show_progress_bar=False)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        n_jobs=1,
        random_state=42,
    )
    model.fit(X_train_emb, y_train, sample_weight=sample_weight)
    return model, enc

def predict_embed_xgb(model, encoder, X_texts):
    X_emb = encoder.encode(X_texts.tolist(), show_progress_bar=False)
    return model.predict(X_emb)

def get_top_dims_xgb(model, n=10):
    # global feature importances
    importances = model.feature_importances_
    idx = np.argsort(importances)[-n:][::-1]
    return [(f"dim_{i}", float(importances[i])) for i in idx]

# ---- Session init ----
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.df = None
    st.session_state.baseline = None
    st.session_state.debiased = None
    st.session_state.model_type = "tfidf_lr"  # or "embed_xgb"

# ---- Sidebar navigation ----
st.sidebar.title("HireFair")
steps = ["1. Load Data", "2. Baseline", "3. Debias", "4. Compare", "5. Report"]
for i, label in enumerate(steps, 1):
    if st.sidebar.button(label, use_container_width=True, type="primary" if i==st.session_state.step else "secondary"):
        st.session_state.step = i
        st.rerun()

# Model selector in sidebar
model_choice = st.sidebar.radio(
    "Model",
    ["TF-IDF + LogisticRegression", "Embedding (MiniLM) + XGBoost"],
    index=0 if st.session_state.model_type=="tfidf_lr" else 1,
    key="model_selector"
)
st.session_state.model_type = "tfidf_lr" if model_choice.startswith("TF-IDF") else "embed_xgb"

# =========================
# STEP 1: Load Data
# =========================
if st.session_state.step == 1:
    st.title("Step 1: Load Resumes")
    st.markdown("""
**Option A:** Upload a CSV with columns `resume_text`, `label` (1=shortlisted, 0=rejected), `group` (demographic).

**Option B:** Upload multiple resume files (PDF, DOCX, TXT) — you'll label them after upload.
""")
    
    tab_csv, tab_files = st.tabs(["📄 CSV Upload", "📂 Resume Files (PDF/DOCX/TXT)"])
    
    with tab_csv:
        file = st.file_uploader("Choose CSV", type=["csv"], key="csv_up")
        if file:
            df = pd.read_csv(file)
            required = {'resume_text','label','group'}
            if not required.issubset(df.columns):
                st.error(f"Missing columns. Need: {required}")
            else:
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows")
                with st.expander("Preview first 10 rows", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Groups: {df['group'].nunique()} • Labels: {df['label'].value_counts().to_dict()}")
    
    with tab_files:
        files = st.file_uploader(
            "Drag & drop or browse", type=["pdf","docx","txt"], 
            accept_multiple_files=True, key="resume_up",
            help="Max 20 files • each truncated to 5 000 chars for preview"
        )
        if files:
            st.info(f"{len(files)} file(s) ready – extracting text…")
            rows = []
            progress = st.progress(0)
            for idx, f in enumerate(files):
                text = extract_text(f)
                rows.append({
                    "file": f.name,
                    "resume_text": text[:5000],
                    "full_text": text,
                    "label": 1,
                    "group": "unknown"
                })
                progress.progress((idx+1)/len(files))
            progress.empty()
            df_files = pd.DataFrame(rows)
            
            st.markdown("### Review & label each resume")
            for i, row in df_files.iterrows():
                with st.container():
                    st.markdown(
                        f"""
                        <div class="file-card">
                          <header>
                            <span class="file-name">{row['file']}</span>
                            <span class="file-meta">{len(row['full_text'])} chars</span>
                          </header>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("Show extracted text (preview)"):
                        st.markdown(f"<div class='preview-box'>{row['resume_text']}</div>", unsafe_allow_html=True)
                    col_l, col_g = st.columns([1,2])
                    label = col_l.selectbox("Label", [0,1], index=1, key=f"lbl_{i}")
                    group = col_g.text_input("Group", value=row['group'], key=f"grp_{i}")
                    df_files.at[i, 'label'] = label
                    df_files.at[i, 'group'] = group
            
            if st.button("✅ Use These Resumes", type="primary", use_container_width=True):
                final_df = df_files[["resume_text","label","group"]].copy()
                st.session_state.df = final_df
                st.success(f"✅ Ready with {len(final_df)} resumes")
                st.dataframe(final_df.head(10), use_container_width=True)

# =========================
# STEP 2: Baseline
# =========================
elif st.session_state.step == 2:
    st.title("Step 2–3: Baseline Model & Bias Audit")
    if st.session_state.df is None:
        st.warning("Upload data first (Step 1).")
    else:
        df = st.session_state.df
        X, y, g = df['resume_text'], df['label'], df['group']
        X_tr, X_te, y_tr, y_te, g_tr, g_te = safe_split(X, y, g)
        
        mtype = st.session_state.model_type
        if mtype == "tfidf_lr":
            model, vec = train_tfidf_lr(X_tr, y_tr)
            preds = model.predict(vec.transform(X_te))
            top_feats = get_top_words_tfidf(model, vec)
            store_vec = vec
        else:  # embed_xgb
            if SentenceTransformer is None or XGBClassifier is None:
                st.error("Embedding+XGBoost requires sentence-transformers and xgboost installed.")
                st.stop()
            model, enc = train_embed_xgb(X_tr, y_tr)
            preds = predict_embed_xgb(model, enc, X_te)
            top_feats = get_top_dims_xgb(model)
            store_vec = enc
        
        metrics = compute_metrics(g_te.values, y_te.values, preds)
        
        st.session_state.baseline = {
            'model_type': mtype,
            'metrics': metrics,
            'model': model,
            'vectorizer': store_vec,
            'X_test': X_te, 'y_test': y_te, 'g_test': g_te,
            'top_feats': top_feats
        }
        
        st.subheader("Group Metrics")
        gs_df = pd.DataFrame(metrics['group_stats'])
        st.dataframe(gs_df.style.format({'selection_rate':'{:.3f}','tpr':'{:.3f}'}), use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        di = metrics['disparate_impact']
        c1.metric("Disparate Impact", f"{di:.3f}", "PASS" if di >= 0.8 else "FAIL",
                  delta_color="normal" if di >= 0.8 else "inverse")
        c2.metric("Equal Opportunity Gap", f"{metrics['equal_opportunity_gap']:.3f}")
        c3.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        
        st.subheader("Top 10 Features")
        tf_df = pd.DataFrame(top_feats, columns=['feature','value'])
        st.dataframe(tf_df.style.format({'value':'{:.4f}'}), use_container_width=True)

# =========================
# STEP 3: Debias
# =========================
elif st.session_state.step == 3:
    st.title("Step 4: Choose Debiasing Method")
    if st.session_state.baseline is None:
        st.warning("Run baseline first (Step 2).")
    else:
        base_type = st.session_state.baseline.get('model_type', 'tfidf_lr')
        method = st.radio(
            "Debiasing method",
            ["term_scrubbing", "reweighting", "counterfactual_augmentation"],
            format_func=lambda x: {
                "term_scrubbing": "Term Scrubbing — mask names, pronouns, gendered words",
                "reweighting": "Reweighting — up-weight underrepresented group-label combos",
                "counterfactual_augmentation": "Counterfactual Augmentation — duplicate with gender terms swapped"
            }[x]
        )
        # disable term_scrubbing for embed_xgb
        if base_type == "embed_xgb" and method == "term_scrubbing":
            st.warning("Term scrubbing only works with TF-IDF model. Switching to reweighting.")
            method = "reweighting"
        
        if st.button("Apply & Retrain", type="primary"):
            df = st.session_state.df
            X, y, g = df['resume_text'], df['label'], df['group']
            X_tr, X_te, y_tr, y_te, g_tr, g_te = safe_split(X, y, g)
            
            # compute sample weights for reweighting
            gl_counts = df.groupby(['group','label']).size()
            total = len(df)
            weights = np.ones(len(X_tr))
            for i, (grp, lbl) in enumerate(zip(g_tr, y_tr)):
                cnt = gl_counts.get((grp, lbl), 1)
                weights[i] = total / (len(gl_counts) * cnt)
            
            if base_type == "tfidf_lr":
                if method == "term_scrubbing":
                    X_tr_db = X_tr.apply(debias_term_scrubbing)
                    model, vec = train_tfidf_lr(X_tr_db, y_tr)
                elif method == "reweighting":
                    model, vec = train_tfidf_lr(X_tr, y_tr, sample_weight=weights)
                else:  # counterfactual
                    X_list, y_list = [], []
                    for txt, lbl in zip(X_tr, y_tr):
                        X_list.append(txt); y_list.append(lbl)
                        swapped = (txt.replace(' he ', ' she ').replace('He ', 'She ')
                                     .replace(' him ', ' her ').replace(' his ', ' her ')
                                     .replace(' she ', ' he ').replace('She ', 'He ')
                                     .replace(' her ', ' him ').replace(' her ', ' his '))
                        X_list.append(swapped); y_list.append(lbl)
                    model, vec = train_tfidf_lr(pd.Series(X_list), pd.Series(y_list))
                preds = model.predict(vec.transform(X_te))
                top_feats = get_top_words_tfidf(model, vec)
                store_vec = vec
            else:  # embed_xgb
                if SentenceTransformer is None or XGBClassifier is None:
                    st.error("Embedding+XGBoost requires sentence-transformers and xgboost installed.")
                    st.stop()
                if method == "reweighting":
                    model, enc = train_embed_xgb(X_tr, y_tr, sample_weight=weights)
                else:  # counterfactual
                    X_list, y_list = [], []
                    for txt, lbl in zip(X_tr, y_tr):
                        X_list.append(txt); y_list.append(lbl)
                        swapped = (txt.replace(' he ', ' she ').replace('He ', 'She ')
                                     .replace(' him ', ' her ').replace(' his ', ' her ')
                                     .replace(' she ', ' he ').replace('She ', 'He ')
                                     .replace(' her ', ' him ').replace(' her ', ' his '))
                        X_list.append(swapped); y_list.append(lbl)
                    model, enc = train_embed_xgb(pd.Series(X_list), pd.Series(y_list))
                preds = predict_embed_xgb(model, enc, X_te)
                top_feats = get_top_dims_xgb(model)
                store_vec = enc
            
            metrics = compute_metrics(g_te.values, y_te.values, preds)
            
            st.session_state.debiased = {
                'model_type': base_type,
                'method': method,
                'metrics': metrics,
                'model': model,
                'vectorizer': store_vec,
                'top_feats': top_feats
            }
            st.success("Debiased model trained. Go to Step 4 to compare.")

# =========================
# STEP 4: Compare
# =========================
elif st.session_state.step == 4:
    st.title("Step 5: Before / After Comparison")
    if st.session_state.baseline is None or st.session_state.debiased is None:
        st.warning("Complete Steps 2 and 3 first.")
    else:
        b = st.session_state.baseline
        d = st.session_state.debiased
        
        st.subheader(f"Method: {d['method'].replace('_',' ').title()} | Model: {b.get('model_type','tfidf_lr')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### BEFORE (Baseline)")
            bm = b['metrics']
            st.metric("Disparate Impact", f"{bm['disparate_impact']:.3f}",
                      "PASS" if bm['disparate_impact']>=0.8 else "FAIL",
                      delta_color="normal" if bm['disparate_impact']>=0.8 else "inverse")
            st.metric("Equal Opportunity Gap", f"{bm['equal_opportunity_gap']:.3f}")
            st.metric("Accuracy", f"{bm['accuracy']:.3f}")
            st.caption("Top 10 Features")
            st.dataframe(pd.DataFrame(b['top_feats'], columns=['feature','value'])
                         .style.format({'value':'{:.4f}'}), use_container_width=True)
        
        with col2:
            st.markdown("### AFTER (Debiased)")
            dm = d['metrics']
            st.metric("Disparate Impact", f"{dm['disparate_impact']:.3f}",
                      "PASS" if dm['disparate_impact']>=0.8 else "FAIL",
                      delta_color="normal" if dm['disparate_impact']>=0.8 else "inverse")
            st.metric("Equal Opportunity Gap", f"{dm['equal_opportunity_gap']:.3f}")
            acc_delta = dm['accuracy'] - bm['accuracy']
            st.metric("Accuracy", f"{dm['accuracy']:.3f}", f"{acc_delta:+.3f}")
            st.caption("Top 10 Features")
            st.dataframe(pd.DataFrame(d['top_feats'], columns=['feature','value'])
                         .style.format({'value':'{:.4f}'}), use_container_width=True)
        
        st.divider()
        st.subheader("Fairness–Accuracy Tradeoff Summary")
        st.markdown(f"""
        | Metric | Before | After | Change |
        |---|---|---|---|
        | Disparate Impact | {bm['disparate_impact']:.3f} | {dm['disparate_impact']:.3f} | {dm['disparate_impact']-bm['disparate_impact']:+.3f} |
        | Equal Opportunity Gap | {bm['equal_opportunity_gap']:.3f} | {dm['equal_opportunity_gap']:.3f} | {dm['equal_opportunity_gap']-bm['equal_opportunity_gap']:+.3f} |
        | Accuracy | {bm['accuracy']:.3f} | {dm['accuracy']:.3f} | {acc_delta:+.3f} |
        """)

# =========================
# STEP 5: Report
# =========================
elif st.session_state.step == 5:
    st.title("Project Report")
    st.markdown("""
## 1. Why resume screening is high-stakes
Resume screening is high-stakes because hiring decisions have significant, long-lasting impacts on individuals' economic mobility, career trajectories, and access to opportunities. Automated screening systems can scale bias to thousands of candidates, creating systemic barriers for historically marginalized groups. Errors or biased patterns can perpetuate workplace inequality and have legal/compliance implications under anti-discrimination laws.

## 2. Fairness metrics: choice and conflicts
We evaluated fairness using:
- **Disparate Impact Ratio** (min selection rate / max selection rate) — Based on the four-fifths rule (0.8 threshold). Chosen as it's widely used in legal contexts (EEOC guidelines) and measures overall selection rate parity.
- **Equal Opportunity Gap** (|TPR_max - TPR_min|) — Measures whether qualified candidates from different groups have equal chances of being selected.

These metrics can conflict: a model can achieve demographic parity while having different true positive rates across groups if base rates differ or the model performs differently. Optimizing for one may hurt the other.

## 3. Accuracy–Fairness Tradeoff
Debiasing techniques (term scrubbing, reweighting, counterfactual augmentation) often reduce model reliance on protected attributes/proxies, improving fairness metrics but may slightly reduce overall accuracy as the model loses predictive signal correlated with bias. The before/after comparison shows this tradeoff explicitly.

## 4. Limits of synthetic data and word-level scrubbing
**Synthetic data limitations:**
- May not capture full complexity of real-world resumes
- Bias patterns may be simplified or exaggerated
- Results may not generalize to actual hiring contexts

**Word-level scrubbing limitations:**
- **Proxy words persist:** Bias leaks through correlated terms (hobbies, club memberships, school names, zip codes)
- **Context matters:** Removing words doesn't address structural biases in how skills/experiences are described or valued
- **Semantic bias remains:** Models associate neutral words with groups based on co-occurrence
- **Over-scrubbing loses information:** Removing relevant terms hurts legitimate predictive power

Effective debiasing requires holistic approaches beyond word-level interventions.
""")
    
    if st.session_state.baseline and st.session_state.debiased:
        b = st.session_state.baseline['metrics']
        d = st.session_state.debiased['metrics']
        st.download_button(
            "Download PDF Report",
            data=generate_pdf_report(b, d, st.session_state.debiased['method']),
            file_name="hirefair_report.pdf",
            mime="application/pdf"
        )

def generate_pdf_report(bm, dm, method):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("HireFair Bias Audit Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Debiasing Method: {method.replace('_',' ').title()}", styles['Heading2']))
    story.append(Spacer(1, 12))
    data = [
        ['Metric', 'Before', 'After', 'Change'],
        ['Disparate Impact', f"{bm['disparate_impact']:.3f}", f"{dm['disparate_impact']:.3f}", f"{dm['disparate_impact']-bm['disparate_impact']:+.3f}"],
        ['Equal Opportunity Gap', f"{bm['equal_opportunity_gap']:.3f}", f"{dm['equal_opportunity_gap']:.3f}", f"{dm['equal_opportunity_gap']-bm['equal_opportunity_gap']:+.3f}"],
        ['Accuracy', f"{bm['accuracy']:.3f}", f"{dm['accuracy']:.3f}", f"{dm['accuracy']-bm['accuracy']:+.3f}"],
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer.read()
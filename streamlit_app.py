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

# Resume parsing
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from docx import Document
except Exception:
    Document = None


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

st.set_page_config(page_title="HireFair — Resume Bias Audit", layout="wide")

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

def train_model(X_train, y_train, vectorizer=None, sample_weight=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
    else:
        X_train_vec = vectorizer.transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train, sample_weight=sample_weight)
    return model, vectorizer

def get_top_words(model, vectorizer, n=10):
    feats = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    idx = np.argsort(coefs)[-n:][::-1]
    return [(feats[i], float(coefs[i])) for i in idx]

# ---- Session init ----
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.df = None
    st.session_state.baseline = None
    st.session_state.debiased = None

# ---- Sidebar navigation ----
st.sidebar.title("HireFair")
steps = ["1. Load Data", "2. Baseline", "3. Debias", "4. Compare", "5. Report"]
for i, label in enumerate(steps, 1):
    if st.sidebar.button(label, use_container_width=True, type="primary" if i==st.session_state.step else "secondary"):
        st.session_state.step = i
        st.rerun()

# =========================
# STEP 1: Load Data
# =========================
if st.session_state.step == 1:
    st.title("Step 1: Load Resumes")
    st.markdown("""
**Option A:** Upload a CSV with columns `resume_text`, `label` (1=shortlisted, 0=rejected), `group` (demographic).

**Option B:** Upload multiple resume files (PDF, DOCX, TXT) — you'll label them after upload.
""")
    
    tab_csv, tab_files = st.tabs(["CSV Upload", "Resume Files (PDF/DOCX/TXT)"])
    
    with tab_csv:
        file = st.file_uploader("Choose CSV", type=["csv"], key="csv_up")
        if file:
            df = pd.read_csv(file)
            required = {'resume_text','label','group'}
            if not required.issubset(df.columns):
                st.error(f"Missing columns. Need: {required}")
            else:
                st.session_state.df = df
                st.success(f"Loaded {len(df)} rows")
                st.dataframe(df.head(10), use_container_width=True)
    
    with tab_files:
        files = st.file_uploader(
            "Upload resume files", type=["pdf","docx","txt"], 
            accept_multiple_files=True, key="resume_up"
        )
        if files:
            st.info(f"{len(files)} files uploaded. Extract text, then assign label & group below.")
            rows = []
            for f in files:
                text = extract_text(f)
                rows.append({"file": f.name, "resume_text": text[:5000], "label": 1, "group": "unknown"})
            df_files = pd.DataFrame(rows)
            
            # Editable table for labels/groups
            edited = st.data_editor(
                df_files[["file","label","group"]],
                column_config={
                    "label": st.column_config.SelectboxColumn("Label", options=[0,1], required=True),
                    "group": st.column_config.TextColumn("Group", required=True),
                },
                hide_index=True,
                use_container_width=True,
                num_rows="fixed"
            )
            # Merge text back
            df_final = df_files.drop(columns=["label","group"]).merge(edited, on="file")
            if st.button("Use These Resumes", type="primary"):
                st.session_state.df = df_final.drop(columns=["file"])
                st.success(f"Ready with {len(df_final)} resumes")
                st.dataframe(st.session_state.df.head(10), use_container_width=True)

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
        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X, y, g, test_size=0.2, random_state=42, stratify=y)
        
        model, vec = train_model(X_tr, y_tr)
        preds = model.predict(vec.transform(X_te))
        metrics = compute_metrics(g_te.values, y_te.values, preds)
        top_words = get_top_words(model, vec)
        
        st.session_state.baseline = {
            'metrics': metrics, 'model': model, 'vectorizer': vec,
            'X_test': X_te, 'y_test': y_te, 'g_test': g_te,
            'top_words': top_words
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
        
        st.subheader("Top 10 Positive Coefficients")
        tw_df = pd.DataFrame(top_words, columns=['word','coefficient'])
        st.dataframe(tw_df.style.format({'coefficient':'{:.4f}'}), use_container_width=True)

# =========================
# STEP 3: Debias
# =========================
elif st.session_state.step == 3:
    st.title("Step 4: Choose Debiasing Method")
    if st.session_state.baseline is None:
        st.warning("Run baseline first (Step 2).")
    else:
        method = st.radio(
            "Debiasing method",
            ["term_scrubbing", "reweighting", "counterfactual_augmentation"],
            format_func=lambda x: {
                "term_scrubbing": "Term Scrubbing — mask names, pronouns, gendered words",
                "reweighting": "Reweighting — up-weight underrepresented group-label combos",
                "counterfactual_augmentation": "Counterfactual Augmentation — duplicate with gender terms swapped"
            }[x]
        )
        
        if st.button("Apply & Retrain", type="primary"):
            df = st.session_state.df
            X, y, g = df['resume_text'], df['label'], df['group']
            X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
                X, y, g, test_size=0.2, random_state=42, stratify=y)
            
            if method == "term_scrubbing":
                X_tr_db = X_tr.apply(debias_term_scrubbing)
                model, vec = train_model(X_tr_db, y_tr)
            elif method == "reweighting":
                # compute weights
                gl_counts = df.groupby(['group','label']).size()
                total = len(df)
                weights = np.ones(len(X_tr))
                for i, (grp, lbl) in enumerate(zip(g_tr, y_tr)):
                    cnt = gl_counts.get((grp, lbl), 1)
                    weights[i] = total / (len(gl_counts) * cnt)
                model, vec = train_model(X_tr, y_tr, sample_weight=weights)
            else:  # counterfactual
                X_list, y_list, g_list = [], [], []
                for txt, lbl, grp in zip(X_tr, y_tr, g_tr):
                    X_list.append(txt); y_list.append(lbl); g_list.append(grp)
                    swapped = (txt.replace(' he ', ' she ').replace('He ', 'She ')
                                 .replace(' him ', ' her ').replace(' his ', ' her ')
                                 .replace(' she ', ' he ').replace('She ', 'He ')
                                 .replace(' her ', ' him ').replace(' her ', ' his '))
                    X_list.append(swapped); y_list.append(lbl); g_list.append(grp)
                model, vec = train_model(pd.Series(X_list), pd.Series(y_list))
            
            preds = model.predict(vec.transform(X_te))
            metrics = compute_metrics(g_te.values, y_te.values, preds)
            top_words = get_top_words(model, vec)
            
            st.session_state.debiased = {
                'method': method,
                'metrics': metrics,
                'model': model,
                'vectorizer': vec,
                'top_words': top_words
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
        
        st.subheader(f"Method: {d['method'].replace('_',' ').title()}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### BEFORE (Baseline)")
            bm = b['metrics']
            st.metric("Disparate Impact", f"{bm['disparate_impact']:.3f}",
                      "PASS" if bm['disparate_impact']>=0.8 else "FAIL",
                      delta_color="normal" if bm['disparate_impact']>=0.8 else "inverse")
            st.metric("Equal Opportunity Gap", f"{bm['equal_opportunity_gap']:.3f}")
            st.metric("Accuracy", f"{bm['accuracy']:.3f}")
            st.caption("Top positive words")
            st.dataframe(pd.DataFrame(b['top_words'], columns=['word','coef'])
                         .style.format({'coef':'{:.4f}'}), use_container_width=True)
        
        with col2:
            st.markdown("### AFTER (Debiased)")
            dm = d['metrics']
            st.metric("Disparate Impact", f"{dm['disparate_impact']:.3f}",
                      "PASS" if dm['disparate_impact']>=0.8 else "FAIL",
                      delta_color="normal" if dm['disparate_impact']>=0.8 else "inverse")
            st.metric("Equal Opportunity Gap", f"{dm['equal_opportunity_gap']:.3f}")
            acc_delta = dm['accuracy'] - bm['accuracy']
            st.metric("Accuracy", f"{dm['accuracy']:.3f}", f"{acc_delta:+.3f}")
            st.caption("Top positive words")
            st.dataframe(pd.DataFrame(d['top_words'], columns=['word','coef'])
                         .style.format({'coef':'{:.4f}'}), use_container_width=True)
        
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
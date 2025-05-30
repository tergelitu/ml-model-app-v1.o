import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# css
st.markdown(
    """
    <style>
    .stApp {
        background-color: #111 !important;
        color: #d0b3ff !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #d8aaff !important;
    }
    .stMarkdown p, .stMarkdown {
        color: #d0b3ff !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111 !important;
    }
    /* Sidebar and widget label text, file name, selectbox, slider, etc. */
    label, .st-af, .st-ag, .st-cq, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz, .st-da, .st-db, .st-dc, .st-dd, .st-de, .st-df, .st-dg, .st-dh, .st-di, .st-dj, .st-dk, .st-dl, .st-dm, .st-dn, .st-do, .st-dp, .st-dq, .st-dr, .st-ds, .st-dt, .st-du, .st-dv, .st-dw, .st-dx, .st-dy, .st-dz, .st-e0, .st-e1, .st-e2, .st-e3, .st-e4, .st-e5, .st-e6, .st-e7, .st-e8, .st-e9, .st-ea, .st-eb, .st-ec, .st-ed, .st-ee, .st-ef, .st-eg, .st-eh, .st-ei, .st-ej, .st-ek, .st-el, .st-em, .st-en, .st-eo, .st-ep, .st-eq, .st-er, .st-es, .st-et, .st-eu, .st-ev, .st-ew, .st-ex, .st-ey, .st-ez, .st-fa, .st-fb, .st-fc, .st-fd, .st-fe, .st-ff, .st-fg, .st-fh, .st-fi, .st-fj, .st-fk, .st-fl, .st-fm, .st-fn, .st-fo, .st-fp, .st-fq, .st-fr, .st-fs, .st-ft, .st-fu, .st-fv, .st-fw, .st-fx, .st-fy, .st-fz, .st-ga, .st-gb, .st-gc, .st-gd, .st-ge, .st-gf, .st-gg, .st-gh, .st-gi, .st-gj, .st-gk, .st-gl, .st-gm, .st-gn, .st-go, .st-gp, .st-gq, .st-gr, .st-gs, .st-gt, .st-gu, .st-gv, .st-gw, .st-gx, .st-gy, .st-gz, .st-ha, .st-hb, .st-hc, .st-hd, .st-he, .st-hf, .st-hg, .st-hh, .st-hi, .st-hj, .st-hk, .st-hl, .st-hm, .st-hn, .st-ho, .st-hp, .st-hq, .st-hr, .st-hs, .st-ht, .st-hu, .st-hv, .st-hw, .st-hx, .st-hy, .st-hz, .st-ia, .st-ib, .st-ic, .st-id, .st-ie, .st-if, .st-ig, .st-ih, .st-ii, .st-ij, .st-ik, .st-il, .st-im, .st-in, .st-io, .st-ip, .st-iq, .st-ir, .st-is, .st-it, .st-iu, .st-iv, .st-iw, .st-ix, .st-iy, .st-iz, .st-ja, .st-jb, .st-jc, .st-jd, .st-je, .st-jf, .st-jg, .st-jh, .st-ji, .st-jj, .st-jk, .st-jl, .st-jm, .st-jn, .st-jo, .st-jp, .st-jq, .st-jr, .st-js, .st-jt, .st-ju, .st-jv, .st-jw, .st-jx, .st-jy, .st-jz, .st-ka, .st-kb, .st-kc, .st-kd, .st-ke, .st-kf, .st-kg, .st-kh, .st-ki, .st-kj, .st-kk, .st-kl, .st-km, .st-kn, .st-ko, .st-kp, .st-kq, .st-kr, .st-ks, .st-kt, .st-ku, .st-kv, .st-kw, .st-kx, .st-ky, .st-kz, .st-la, .st-lb, .st-lc, .st-ld, .st-le, .st-lf, .st-lg, .st-lh, .st-li, .st-lj, .st-lk, .st-ll, .st-lm, .st-ln, .st-lo, .st-lp, .st-lq, .st-lr, .st-ls, .st-lt, .st-lu, .st-lv, .st-lw, .st-lx, .st-ly, .st-lz, .st-ma, .st-mb, .st-mc, .st-md, .st-me, .st-mf, .st-mg, .st-mh, .st-mi, .st-mj, .st-mk, .st-ml, .st-mm, .st-mn, .st-mo, .st-mp, .st-mq, .st-mr, .st-ms, .st-mt, .st-mu, .st-mv, .st-mw, .st-mx, .st-my, .st-mz, .st-na, .st-nb, .st-nc, .st-nd, .st-ne, .st-nf, .st-ng, .st-nh, .st-ni, .st-nj, .st-nk, .st-nl, .st-nm, .st-nn, .st-no, .st-np, .st-nq, .st-nr, .st-ns, .st-nt, .st-nu, .st-nv, .st-nw, .st-nx, .st-ny, .st-nz, .st-oa, .st-ob, .st-oc, .st-od, .st-oe, .st-of, .st-og, .st-oh, .st-oi, .st-oj, .st-ok, .st-ol, .st-om, .st-on, .st-oo, .st-op, .st-oq, .st-or, .st-os, .st-ot, .st-ou, .st-ov, .st-ow, .st-ox, .st-oy, .st-oz, .st-pa, .st-pb, .st-pc, .st-pd, .st-pe, .st-pf, .st-pg, .st-ph, .st-pi, .st-pj, .st-pk, .st-pl, .st-pm, .st-pn, .st-po, .st-pp, .st-pq, .st-pr, .st-ps, .st-pt, .st-pu, .st-pv, .st-pw, .st-px, .st-py, .st-pz, .st-qa, .st-qb, .st-qc, .st-qd, .st-qe, .st-qf, .st-qg, .st-qh, .st-qi, .st-qj, .st-qk, .st-ql, .st-qm, .st-qn, .st-qo, .st-qp, .st-qq, .st-qr, .st-qs, .st-qt, .st-qu, .st-qv, .st-qw, .st-qx, .st-qy, .st-qz, .st-ra, .st-rb, .st-rc, .st-rd, .st-re, .st-rf, .st-rg, .st-rh, .st-ri, .st-rj, .st-rk, .st-rl, .st-rm, .st-rn, .st-ro, .st-rp, .st-rq, .st-rr, .st-rs, .st-rt, .st-ru, .st-rv, .st-rw, .st-rx, .st-ry, .st-rz, .st-sa, .st-sb, .st-sc, .st-sd, .st-se, .st-sf, .st-sg, .st-sh, .st-si, .st-sj, .st-sk, .st-sl, .st-sm, .st-sn, .st-so, .st-sp, .st-sq, .st-sr, .st-ss, .st-st, .st-su, .st-sv, .st-sw, .st-sx, .st-sy, .st-sz, .st-ta, .st-tb, .st-tc, .st-td, .st-te, .st-tf, .st-tg, .st-th, .st-ti, .st-tj, .st-tk, .st-tl, .st-tm, .st-tn, .st-to, .st-tp, .st-tq, .st-tr, .st-ts, .st-tt, .st-tu, .st-tv, .st-tw, .st-tx, .st-ty, .st-tz, .st-ua, .st-ub, .st-uc, .st-ud, .st-ue, .st-uf, .st-ug, .st-uh, .st-ui, .st-uj, .st-uk, .st-ul, .st-um, .st-un, .st-uo, .st-up, .st-uq, .st-ur, .st-us, .st-ut, .st-uu, .st-uv, .st-uw, .st-ux, .st-uy, .st-uz, .st-va, .st-vb, .st-vc, .st-vd, .st-ve, .st-vf, .st-vg, .st-vh, .st-vi, .st-vj, .st-vk, .st-vl, .st-vm, .st-vn, .st-vo, .st-vp, .st-vq, .st-vr, .st-vs, .st-vt, .st-vu, .st-vv, .st-vw, .st-vx, .st-vy, .st-vz, .st-wa, .st-wb, .st-wc, .st-wd, .st-we, .st-wf, .st-wg, .st-wh, .st-wi, .st-wj, .st-wk, .st-wl, .st-wm, .st-wn, .st-wo, .st-wp, .st-wq, .st-wr, .st-ws, .st-wt, .st-wu, .st-wv, .st-ww, .st-wx, .st-wy, .st-wz, .st-xa, .st-xb, .st-xc, .st-xd, .st-xe, .st-xf, .st-xg, .st-xh, .st-xi, .st-xj, .st-xk, .st-xl, .st-xm, .st-xn, .st-xo, .st-xp, .st-xq, .st-xr, .st-xs, .st-xt, .st-xu, .st-xv, .st-xw, .st-xx, .st-xy, .st-xz, .st-ya, .st-yb, .st-yc, .st-yd, .st-ye, .st-yf, .st-yg, .st-yh, .st-yi, .st-yj, .st-yk, .st-yl, .st-ym, .st-yn, .st-yo, .st-yp, .st-yq, .st-yr, .st-ys, .st-yt, .st-yu, .st-yv, .st-yw, .st-yx, .st-yy, .st-yz, .st-za, .st-zb, .st-zc, .st-zd, .st-ze, .st-zf, .st-zg, .st-zh, .st-zi, .st-zj, .st-zk, .st-zl, .st-zm, .st-zn, .st-zo, .st-zp, .st-zq, .st-zr, .st-zs, .st-zt, .st-zu, .st-zv, .st-zw, .st-zx, .st-zy, .st-zz {
        color: #d8aaff !important;
    }
    /* File name in uploader */
    .st-dq, .st-emotion-cache-1cypcdb {
        color: #d8aaff !important;
    }
    /* Expander/section frame border */
    .stExpander, .st-emotion-cache-1v0mbdj, .st-emotion-cache-1vzeuhh {
        border: 2px solid #d8aaff !important;
        box-shadow: 0 0 8px #7209b744 !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# main section
st.markdown('<h1 style="color:#d8aaff;">ML Model App</h1>', unsafe_allow_html=True)

st.markdown(
    """
    <iframe src='https://my.spline.design/interactivespherescopy-q3PqOME4PbnxqZ3vogUqkMhp/' 
    frameborder='0' width='100%' height='300%'></iframe>
    """,
    unsafe_allow_html=True
)


# data upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    st.warning("Please upload a CSV or Excel file.")
    st.stop()

# target column selection
with st.expander("Data"):
    st.write("**Select your target column:**")
    y_col = st.selectbox("Target column (y)", df.columns)
    st.write("**Data Preview:**")
    st.dataframe(df)

# descriptive analysis
with st.expander("Descriptive Analysis"):
    st.write("**Descriptive statistics for all features:**")
    st.dataframe(df.describe(include='all').T)

    st.write("**Feature histograms:**")
    feature = st.selectbox("Select feature for histogram", [col for col in df.columns if col != y_col])
    if pd.api.types.is_numeric_dtype(df[feature]):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[feature], marker_color='royalblue'))
        fig.update_layout(title=f"Histogram of {feature}", xaxis_title=feature, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure()
        counts = df[feature].value_counts()
        fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts.values, marker_color='royalblue'))
        fig.update_layout(title=f"Bar plot of {feature}", xaxis_title=feature, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# data cleaning
with st.expander("Data Cleaning"):
    st.write("**Null percentage of features:**")
    null_percentage = df.isnull().mean() * 100

    cutoff = st.slider("Null value cutoff (%)", min_value=0, max_value=100, value=50, step=1)
    st.write(f"Columns with null percentage > {cutoff}% will be dropped.")

    fig = go.Figure(data=[
        go.Bar(
            x=null_percentage.index,
            y=null_percentage.values,
            marker=dict(color='darkblue'),
        )
    ])
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(null_percentage) - 0.5,
        y0=cutoff,
        y1=cutoff,
        line=dict(color="red", width=2, dash="dash"),
    )
    fig.update_layout(
        title="Null percentage of features",
        xaxis_title="Columns",
        yaxis_title="Null Percentage (%)",
        xaxis=dict(tickangle=45),
        height=500,
        width=900,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    columns_to_drop = null_percentage[null_percentage > cutoff].index
    columns_to_keep = null_percentage[null_percentage <= cutoff].index
    columns_to_drop = [col for col in columns_to_drop if col not in ['ID', 'TARGET']]
    df_clean = df.drop(columns=columns_to_drop)

    # st.write(f"**Dropped columns ({len(columns_to_drop)}):** {list(columns_to_drop)}")
    # st.write(f"**Kept columns ({len(columns_to_keep)}):** {list(columns_to_keep)}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    dropped_numeric = [col for col in columns_to_drop if col in numeric_cols]
    kept_numeric = [col for col in columns_to_keep if col in numeric_cols]

    if dropped_numeric or kept_numeric:
        fig_box = go.Figure()
        for col in kept_numeric:
            fig_box.add_trace(go.Box(y=df[col], name=f"Kept: {col}", marker_color='royalblue'))
        for col in dropped_numeric:
            fig_box.add_trace(go.Box(y=df[col], name=f"Dropped: {col}", marker_color='red'))
        fig_box.update_layout(
            title="Boxplot: Kept vs Dropped Numeric Columns",
            yaxis_title="Value",
            boxmode='group',
            height=500,
            width=900
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No numeric columns to show in boxplot.")

#feature selection
with st.expander("Feature Selection"):
    st.write("**Feature selection using feature importance and correlation filtering.**")

    id_col = st.selectbox("Select ID column", df_clean.columns)
    target_col = st.selectbox("Select target column", [col for col in df_clean.columns if col != id_col])

    df_work = df_clean.copy()

    threshold = st.slider("Correlation threshold for dropping features", min_value=0.5, max_value=1.0, value=0.9, step=0.01)
    top_percent = st.slider("Top percent of features to keep", min_value=1, max_value=100, value=20, step=1)

    def clean_feature_names(df):
        df.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '')
                         .replace(' ', '_').replace('"', '').replace("'", '')
                         .replace('{', '').replace('}', '').replace(':', '') for col in df.columns]
        return df

    def remove_high_corr_features(X, importance_dict, threshold=0.9):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            for row in upper.index:
                if upper.loc[row, col] > threshold:
                    if importance_dict.get(row, 0) > importance_dict.get(col, 0):
                        to_drop.add(col)
                    else:
                        to_drop.add(row)
        return X.drop(columns=to_drop)

    def get_feature_importance(model, feature_names, model_name):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        importance_df['Importance'] = importance_df['Importance'].apply(lambda x: f"{x:.4f}")
        st.write(f"🔹 Feature Importance for {model_name}:")
        st.dataframe(importance_df)
        return importance_df

    def select_top_features(model, X_train, y_train, model_name, threshold, top_percent):
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        importance_dict = dict(zip(X_train.columns, importances))

        X_filtered = remove_high_corr_features(X_train, importance_dict, threshold=threshold)

        model.fit(X_filtered, y_train)
        importances = model.feature_importances_
        feature_df = pd.DataFrame({'Feature': X_filtered.columns, 'Importance': importances})

        quantile_threshold = feature_df['Importance'].quantile(1 - top_percent / 100)
        selected_features_df = feature_df[feature_df['Importance'] >= quantile_threshold].sort_values(by='Importance', ascending=False)

        return selected_features_df['Feature'].values

    df_raw = clean_feature_names(df_work)
    X = df_raw.drop(columns=[id_col, target_col])
    y = df_raw[target_col]

    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42), 
    }

    selected_features_dict = {}
    feature_importance_dict = {}

    for name, model in models.items():
        selected_features = select_top_features(model, X_train, y_train, name, threshold, top_percent)
        selected_features_dict[name] = selected_features
        fitted_model = model.fit(X_train[selected_features], y_train)
        feature_importance_dict[name] = get_feature_importance(fitted_model, selected_features, name)

    feature_stats = {}
    for name, importance_df in feature_importance_dict.items():
        imp = importance_df['Importance'].astype(float)
        norm_importance = (imp - imp.min()) / (imp.max() - imp.min() + 1e-9)
        for feature, norm_imp in zip(importance_df['Feature'], norm_importance):
            if feature not in feature_stats:
                feature_stats[feature] = {'count': 0, 'total_importance': 0.0, 'models': []}
            feature_stats[feature]['count'] += 1
            feature_stats[feature]['total_importance'] += norm_imp
            feature_stats[feature]['models'].append(name)

    feature_ranking = pd.DataFrame([
        {
            'Feature': feat,
            'Models_Selected': stat['count'],
            'Avg_Normalized_Importance': stat['total_importance'] / stat['count']
        }
        for feat, stat in feature_stats.items()
    ])

    feature_ranking = feature_ranking.sort_values(
        by=['Models_Selected', 'Avg_Normalized_Importance'], ascending=[False, False]
    )

    st.write("Feature Ranking (Top 30)")
    st.dataframe(feature_ranking.head(30))

    suspicious_features = []
    for name, importance_df in feature_importance_dict.items():
        imp = importance_df['Importance'].astype(float)
        norm_importance = (imp - imp.min()) / (imp.max() - imp.min() + 1e-9)
        for feature, norm_imp in zip(importance_df['Feature'], norm_importance):
            if norm_imp > 0.95:
                suspicious_features.append((feature, name, norm_imp))

    if suspicious_features:
        st.warning("⚠️ Suspicious features with very high normalized importance (>0.95):")
        for feat, model, norm_imp in suspicious_features:
            st.write(f"Feature: {feat} | Model: {model} | Normalized Importance: {norm_imp:.3f}")
    else:
        st.info("No suspicious features detected with normalized importance > 0.95.")

#modeling
with st.expander("Modeling"):

    sus_feature_names = set([feat for feat, _, _ in suspicious_features])
    for name in selected_features_dict:
        filtered = [f for f in selected_features_dict[name] if f not in sus_feature_names]
        selected_features_dict[name] = filtered

    def evaluate_model(model, X_train, X_test, y_train, y_test, selected_features):
        model.fit(X_train[selected_features], y_train)
        y_train_pred = model.predict_proba(X_train[selected_features])[:, 1]
        y_test_pred = model.predict_proba(X_test[selected_features])[:, 1]
        y_test_pred_label = model.predict(X_test[selected_features])
        return {
            'Train ROC AUC': roc_auc_score(y_train, y_train_pred),
            'Test ROC AUC': roc_auc_score(y_test, y_test_pred),
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'y_test_pred_label': y_test_pred_label
        }

    results_dict = {}

    fig_train_roc = go.Figure()
    for name, model in models.items():
        features = selected_features_dict[name]
        if not features:
            st.warning(f"No features left for {name} after removing suspicious features.")
            continue
        results = evaluate_model(model, X_train, X_test, y_train, y_test, features)
        results_dict[name] = results
        fpr, tpr, _ = roc_curve(y_train, results['y_train_pred'])
        roc_auc = auc(fpr, tpr)
        fig_train_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={roc_auc:.3f})'))
    fig_train_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='navy'), showlegend=False
    ))
    fig_train_roc.update_layout(
        title="Train ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800, height=500, legend=dict(x=0.7, y=0.05), template="plotly_white"
    )
    st.plotly_chart(fig_train_roc, use_container_width=True)

    fig_test_roc = go.Figure()
    for name, results in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, results['y_test_pred'])
        roc_auc = auc(fpr, tpr)
        fig_test_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={roc_auc:.3f})'))
    fig_test_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='navy'), showlegend=False
    ))
    fig_test_roc.update_layout(
        title="Test ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800, height=500, legend=dict(x=0.7, y=0.05), template="plotly_white"
    )
    st.plotly_chart(fig_test_roc, use_container_width=True)

    if results_dict:
        model_names = list(results_dict.keys())
        train_roc = [results_dict[n]['Train ROC AUC'] for n in model_names]
        test_roc = [results_dict[n]['Test ROC AUC'] for n in model_names]
        power_loss = [tr - te for tr, te in zip(train_roc, test_roc)]

        fig_loss = go.Figure(data=[
            go.Bar(x=model_names, y=power_loss, marker_color='crimson')
        ])
        fig_loss.update_layout(
            title="Power Loss (Train ROC - Test ROC) for Each Model",
            yaxis_title="Power Loss",
            xaxis_title="Model",
            width=700,
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    st.subheader("Model Performance Metrics")
    for name, results in results_dict.items():
        y_test_pred_label = results['y_test_pred_label']
        acc = accuracy_score(y_test, y_test_pred_label)
        prec = precision_score(y_test, y_test_pred_label, zero_division=0)
        rec = recall_score(y_test, y_test_pred_label, zero_division=0)
        f1 = f1_score(y_test, y_test_pred_label, zero_division=0)
        cm = confusion_matrix(y_test, y_test_pred_label)
        st.markdown(f"**{name}**")
        st.write(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1-score: {f1:.3f}")
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))



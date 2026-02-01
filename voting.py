import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Core
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import r2_score, mean_absolute_error

# Ensemble & Models
from sklearn.ensemble import VotingClassifier, VotingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Ensemble Diagnostic Lab", layout="wide", page_icon="🔬")

# --- SIDEBAR: GLOBAL MODE SELECTOR ---
st.sidebar.title("🚀 Lab Selection")
app_mode = st.sidebar.radio("Choose Analysis Type", ["Classification", "Regression"])

# ---------------------------------------------------------
# SECTION 1: CLASSIFICATION CODE
# ---------------------------------------------------------
if app_mode == "Classification":
    def generate_xor():
        rng = np.random.RandomState(42)
        X = rng.randn(400, 2)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
        return X, y

    def generate_spirals():
        n_points = 250
        theta = np.sqrt(np.random.rand(n_points)) * 2 * np.pi
        r_a = 2 * theta + np.pi
        data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
        x_a = data_a + np.random.randn(n_points, 2)
        r_b = -2 * theta - np.pi
        data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
        x_b = data_b + np.random.randn(n_points, 2)
        X = np.vstack([x_a, x_b])
        y = np.hstack([np.zeros(n_points), np.ones(n_points)])
        return X, y

    st.sidebar.title("🕹️ Voting Classifier")
    st.sidebar.subheader("1. Dataset")
    ds_map = {
        "Two Spirals": generate_spirals,
        "XOR": generate_xor,
        "Concentric Circles": lambda: datasets.make_circles(n_samples=400, noise=0.1, factor=0.4, random_state=42),
        "Interlocking Moons": lambda: datasets.make_moons(n_samples=400, noise=0.15, random_state=42),
        "Linearly Separable": lambda: datasets.make_blobs(n_samples=400, centers=2, cluster_std=1.2, random_state=42)
    }
    ds_choice = st.sidebar.selectbox("Select Challenge", list(ds_map.keys()))

    st.sidebar.subheader("2. Active Estimators")
    active_models = []
    if st.sidebar.checkbox("K-Nearest Neighbors (KNN)", True): active_models.append("knn")
    if st.sidebar.checkbox("Logistic Regression (LR)", True): active_models.append("lr")
    if st.sidebar.checkbox("Gaussian Naive Bayes (GNB)", True): active_models.append("gnb")
    if st.sidebar.checkbox("Support Vector Machine (SVM)", True): active_models.append("svm")
    if st.sidebar.checkbox("Random Forest (RF)", True): active_models.append("rf")

    with st.sidebar.expander("🛠️ VotingClassifier Hyperparameters", expanded=True):
        v_mode = st.selectbox("Voting Type", ["soft", "hard"])
        n_jobs = st.number_input("n_jobs", value=-1)
        flatten_tf = st.checkbox("flatten_transform", value=True)
        verbose = st.checkbox("verbose", value=False)
        st.markdown("**Manual Weights**")
        weights = [st.slider(f"Weight: {m.upper()}", 0.0, 5.0, 1.0, 0.1) for m in active_models]

    X_raw, y_raw = ds_map[ds_choice]()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=0.25, random_state=42)

    base_estimators_dict = {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'lr': LogisticRegression(),
        'gnb': GaussianNB(),
        'svm': SVC(probability=(v_mode == "soft"), kernel='rbf'),
        'rf': RandomForestClassifier(n_estimators=100, max_depth=5)
    }

    st.title("🔬 Advanced Ensemble Diagnostic Lab")
    if len(active_models) == 0:
        st.error("🚨 Please select at least one estimator in the sidebar.")
    else:
        current_estimators = [(m, base_estimators_dict[m]) for m in active_models]
        if len(active_models) > 1:
            model_to_plot = VotingClassifier(estimators=current_estimators, voting=v_mode, weights=weights,
                                             n_jobs=int(n_jobs), flatten_transform=flatten_tf, verbose=verbose)
        else:
            model_to_plot = base_estimators_dict[active_models[0]]

        model_to_plot.fit(X_train, y_train)
        y_pred = model_to_plot.predict(X_test)

        st.subheader("🏁 Estimators Performance")
        score_cols = st.columns(len(active_models) + 1)
        individual_accuracies = {}
        for i, (name, clf) in enumerate(current_estimators):
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            individual_accuracies[name] = acc
            score_cols[i].metric(f"{name.upper()} Acc", f"{acc:.1%}")

        final_acc = accuracy_score(y_test, y_pred)
        score_cols[-1].metric("ENSEMBLE Acc", f"{final_acc:.1%}")
        st.markdown("---")

        tab_boundary, tab_metrics = st.tabs(["🌐 Spatial Logic", "📊 Performance Metrics"])
        with tab_boundary:
            h = .04
            x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
            y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            grid_points = np.c_[xx.ravel(), yy.ravel()]

            st.subheader("Collective Decision Boundary")
            fig_big, ax_big = plt.subplots(figsize=(10, 5))
            Z_final = model_to_plot.predict(grid_points).reshape(xx.shape)
            ax_big.contourf(xx, yy, Z_final, alpha=0.5, cmap='RdYlBu')
            ax_big.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_raw, edgecolors='k', cmap='RdYlBu', s=30)
            st.pyplot(fig_big)

            st.markdown("### Individual Estimator Logic")
            n_models = len(active_models)
            fig_small, axes_small = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5))
            if n_models == 1: axes_small = [axes_small]
            for i, (name, clf) in enumerate(current_estimators):
                Z_ind = clf.predict(grid_points).reshape(xx.shape)
                axes_small[i].contourf(xx, yy, Z_ind, alpha=0.3, cmap='RdYlBu')
                axes_small[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_raw, edgecolors='k', cmap='RdYlBu', s=10, alpha=0.4)
                axes_small[i].set_title(f"{name.upper()}")
                axes_small[i].axis('off')
            st.pyplot(fig_small)

        with tab_metrics:
            viz_col, report_col = st.columns([1, 1])
            with viz_col:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                st.pyplot(fig_cm)
            with report_col:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))

# ---------------------------------------------------------
# SECTION 2: REGRESSION CODE (FRIEDMAN FIXED)
# ---------------------------------------------------------
else:
    st.title("📉 Ensemble Regression Diagnostic Lab")
    
    st.sidebar.subheader("1. Dataset Selection")
    def get_reg_data(choice):
        n_samples = 400
        if choice == "Friedman Regression":
            return datasets.make_friedman1(n_samples=n_samples, n_features=5, noise=1.0, random_state=42)
        elif choice == "Diabetes (Toy)":
            return datasets.load_diabetes(return_X_y=True)
        
        X = np.sort(np.random.rand(n_samples, 1) * 10, axis=0)
        if choice == "Linear Trend":
            y = 2.5 * X.flatten() + np.random.normal(0, 2, n_samples)
        elif choice == "Quadratic (Non-Linear)":
            y = 0.5 * (X.flatten() - 5)**2 + np.random.normal(0, 1, n_samples)
        elif choice == "Sinusoidal (Wave)":
            y = np.sin(X.flatten()) * 5 + np.random.normal(0, 0.5, n_samples)
        return X, y

    reg_ds = st.sidebar.selectbox("Select Regression Challenge", 
                                  ["Linear Trend", "Quadratic (Non-Linear)", "Sinusoidal (Wave)", "Friedman Regression", "Diabetes (Toy)"])
    
    st.sidebar.subheader("2. Active Regressors")
    active_regs = []
    if st.sidebar.checkbox("Linear Regression (LR)", True): active_regs.append("lr")
    if st.sidebar.checkbox("SVR (RBF Kernel)", True): active_regs.append("svr")
    if st.sidebar.checkbox("Decision Tree (DT)", True): active_regs.append("dt")

    st.sidebar.markdown("**Manual Weights**")
    reg_weights = [st.sidebar.slider(f"Weight: {m.upper()}", 0.0, 5.0, 1.0, 0.1) for m in active_regs]

    X_reg, y_reg = get_reg_data(reg_ds)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_dict =

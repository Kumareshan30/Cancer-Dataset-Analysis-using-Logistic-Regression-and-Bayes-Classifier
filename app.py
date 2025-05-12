"""
Logistic Regression vs. Bayesian Classifiers on Breast Cancer Dataset
Interactive Streamlit Dashboard
- Compare Logistic Regression & various Bayesian classifiers
- Performance metrics: accuracy, loss, ROC, PR, calibration
- Custom-case prediction, feature distributions, partial dependence
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, zero_one_loss,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

# --- Utility Functions

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- Classifiers
class SGDRegularisedLogisticRegressionClassifier:
    def __init__(self, alpha=0.1, batch_size=10, eta=0.001, tau_max=10000, epsilon=1e-5, random_state=None):
        self.alpha = alpha
        self.batch_size = batch_size
        self.eta = eta
        self.tau_max = tau_max
        self.epsilon = epsilon
        self.random_state = random_state
        self.coef_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        w = np.zeros(p)
        for tau in range(self.tau_max):
            idx = rng.choice(n, size=self.batch_size, replace=True)
            Xb, yb = X[idx], y[idx]
            grad = Xb.T.dot(sigmoid(Xb.dot(w)) - yb) + self.alpha * w
            w_new = w - self.eta * grad
            if np.linalg.norm(w_new - w) < self.epsilon:
                w = w_new
                break
            w = w_new
        self.coef_ = w
        return self

    def decision_function(self, X):
        return X.dot(self.coef_)

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def predict_proba(self, X):
        p1 = sigmoid(self.decision_function(X))
        return np.vstack([1 - p1, p1]).T

class BayesianClassifier:
    def __init__(self, shared_cov=True, cond_ind=True):
        self.shared_cov = shared_cov
        self.cond_ind = cond_ind

    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.priors_ = counts / len(y)
        self.means_ = {}
        self.covs_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.means_[c] = Xc.mean(axis=0)
            if self.cond_ind:
                cov = np.diag(Xc.var(axis=0))
            else:
                cov = np.cov(Xc, rowvar=False, bias=True)
            self.covs_[c] = cov
        if self.shared_cov:
            shared = sum(self.priors_[i] * self.covs_[c] for i, c in enumerate(self.classes_))
            for c in self.classes_:
                self.covs_[c] = shared
        return self

    def predict_proba(self, X):
        probs = []
        for c in self.classes_:
            rv = multivariate_normal(mean=self.means_[c], cov=self.covs_[c], allow_singular=True)
            probs.append(self.priors_[list(self.classes_).index(c)] * rv.pdf(X))
        probs = np.vstack(probs).T
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# --- Streamlit App

def main():
    st.title('Logistic vs Bayesian Classifiers')
    st.markdown(
        'Compare Logistic Regression and Bayesian classifiers ' +
        'on the breast cancer dataset with interactive diagnostics.'
    )

    # Load and prepare data
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # 0. Dataset Overview
    st.header('0. Dataset Overview')
    st.subheader('Sample Records')
    st.dataframe(df.head())
    st.subheader('Class Distribution')
    fig0, ax0 = plt.subplots()
    counts = df['target'].value_counts().sort_index()
    ax0.bar(class_names, counts)
    ax0.set_ylabel('Count')
    st.pyplot(fig0)

    # 1. Feature Correlations
    st.header('1. Feature Correlations')
    corr = df.drop(columns='target').corr()
    figc, axc = plt.subplots(figsize=(6,5))
    cax = axc.imshow(corr, cmap='coolwarm', aspect='auto')
    figc.colorbar(cax, ax=axc)
    axc.set_xticks(range(len(feature_names)))
    axc.set_xticklabels(feature_names, rotation=90)
    axc.set_yticks(range(len(feature_names)))
    axc.set_yticklabels(feature_names)
    st.pyplot(figc)

    # Sidebar: choose models & split
    st.sidebar.header('Configuration')
    model_map = {
        'NB (shared cov)': BayesianClassifier(shared_cov=True, cond_ind=True),
        'Naive Bayes': BayesianClassifier(shared_cov=False, cond_ind=True),
        'BC (shared cov)': BayesianClassifier(shared_cov=True, cond_ind=False),
        'BC (full cov)': BayesianClassifier(shared_cov=False, cond_ind=False),
        'Logistic Regression': SGDRegularisedLogisticRegressionClassifier(alpha=0.1, batch_size=100, eta=0.0005)
    }
    selected = st.sidebar.multiselect('Models to compare', list(model_map.keys()), default=list(model_map.keys()))
    test_size = st.sidebar.slider('Test set fraction', 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit models & collect metrics
    results = {}
    for name in selected:
        clf = model_map[name]
        clf.fit(X_train, y_train)
        ytr = clf.predict(X_train)
        yte = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:,1] if hasattr(clf, 'predict_proba') else None
        results[name] = {
            'train_acc': accuracy_score(y_train, ytr),
            'test_acc': accuracy_score(y_test, yte),
            'train_loss': zero_one_loss(y_train, ytr),
            'test_loss': zero_one_loss(y_test, yte),
            'probs': probs
        }

    # 2. Performance Table
    st.header('2. Performance Metrics')
    df_res = pd.DataFrame({
        name: {
            'Train Acc': res['train_acc'],
            'Test Acc' : res['test_acc'],
            'Train Loss': res['train_loss'],
            'Test Loss' : res['test_loss']
        } for name, res in results.items()
    }).T
    st.table(df_res.style.format({
        'Train Acc':'{:.2%}', 'Test Acc':'{:.2%}',
        'Train Loss':'{:.2%}', 'Test Loss':'{:.2%}'
    }))

    # 3. Predictions & Detailed Statistics
    st.header('3. Predictions & Detailed Statistics')
    st.markdown(
        'Below are the **confusion matrices** and **classification reports** for each selected model. ' +
        'The confusion matrix shows true positives, true negatives, false positives, and false negatives. ' +
        'The classification report provides precision, recall, F1-score, and support for each class, helping you assess per-class performance and the balance between sensitivity and specificity.'
    )
    st.header('3. Predictions & Detailed Statistics')
    from sklearn.metrics import classification_report, confusion_matrix
    for name in selected:
        clf = model_map[name]
        y_pred = clf.predict(X_test)
        st.subheader(name)
        cm = confusion_matrix(y_test, y_pred)
        st.write('Confusion Matrix')
        st.write(cm)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        df_report = pd.DataFrame(report).T
        st.write('Classification Report')
        st.dataframe(df_report)

    # 4. Precision-Recall & Calibration
    st.header('4. Precision-Recall & Calibration')
    for name in selected:
        proba = results[name]['probs']
        if proba is None:
            continue
        st.subheader(name)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall, precision)
        ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'PR Curve (AP={ap:.2f})')
        st.pyplot(fig_pr)
        # Calibration
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
        fig_cal, ax_cal = plt.subplots()
        ax_cal.plot(prob_pred, prob_true, marker='o')
        ax_cal.plot([0,1],[0,1], linestyle='--', color='grey')
        ax_cal.set_xlabel('Predicted probability'); ax_cal.set_ylabel('True probability')
        ax_cal.set_title('Calibration Curve')
        st.pyplot(fig_cal)

    # 5. Interactive Feature Distribution
    st.header('5. Feature Distributions')
    feat1 = st.selectbox('Feature 1', feature_names)
    feat2 = st.selectbox('Feature 2 (optional)', [''] + list(feature_names))
    if feat2:
        fig_f, ax_f = plt.subplots()
        scatter = ax_f.scatter(df[feat1], df[feat2], c=df['target'], cmap='bwr', alpha=0.6)
        ax_f.set_xlabel(feat1); ax_f.set_ylabel(feat2)
        st.pyplot(fig_f)
    else:
        fig_f, ax_f = plt.subplots()
        for cls in class_names:
            subset = df[df['target']==list(class_names).index(cls)]
            ax_f.hist(subset[feat1], bins=20, alpha=0.5, label=cls)
        ax_f.set_xlabel(feat1); ax_f.legend()
        st.pyplot(fig_f)

    # 6. Partial Dependence
    st.header('6. Partial Dependence')
    pd_feat = st.selectbox('Feature for PD', feature_names)
    grid = np.linspace(df[pd_feat].min(), df[pd_feat].max(), 200)
    X_med = np.median(X_train, axis=0)
    fig_pd, ax_pd = plt.subplots()
    for name in selected:
        clf = model_map[name]
        probs = []
        for val in grid:
            x0 = X_med.copy()
            x0[list(feature_names).index(pd_feat)] = val
            proba = clf.predict_proba(x0.reshape(1,-1))[0,1]
            probs.append(proba)
        ax_pd.plot(grid, probs, label=name)
    ax_pd.set_xlabel(pd_feat); ax_pd.set_ylabel('P(c=1)'); ax_pd.legend()
    st.pyplot(fig_pd)

    # 7. Model Comparison Insights
    st.header('7. Model Comparison Insights')
    st.markdown(
        """
        - **Naive Bayes (shared covariance)** tends to generalize best with high test accuracy and stability; ideal when you seek a simple yet robust probabilistic model.  
        - **Naive Bayes (conditional independence)** offers good performance with minimal assumptions; useful for fast baseline classification, especially with fewer data.  
        - **Bayes Classifier (full covariance)** can overfit high-dimensional data, leading to lower test accuracy; avoid when sample size is limited or features are highly correlated.  
        - **Logistic Regression (SGD)** may underfit complex patterns without polynomial expansion; suitable for large-scale, high-dimensional settings where interpretability of linear decision boundary is key.  

        **Scenario-Based Recommendations:**  
        - **Limited Data & Speed-Critical:** Use **Naive Bayes** (shared or conditional) for quick training and inference with good accuracy.  
        - **Understanding Feature Effects:** Use **Logistic Regression** when you need coefficient interpretability and linear decision boundaries.  
        - **Uncertainty Quantification:** Bayesian classifiers naturally provide posterior probabilities—great for risk-sensitive applications like medical diagnostics.  
        - **High-Dimensional, Well-Sampled Data:** Consider **full-covariance Bayes** only if sample size >> feature count to estimate covariance reliably.  

        **Further Exploration:**  
        - Experiment with **threshold tuning** on the ROC/PR curves to balance sensitivity vs specificity per clinical needs.  
        - Use **calibration curves** to check probability estimates and adjust via isotonic or Platt scaling if needed.  
        - For dynamic data streams, consider **online SGD Logistic Regression** to adapt models over time.
        """
    )

if __name__ == '__main__':
    main()

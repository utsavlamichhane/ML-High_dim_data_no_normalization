###loading all the libs


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import glob
import joblib
import shap

## for RMse, easy peasy funct

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

DATA_PATTERN = 'preprocessed_ASV_level*.csv'     
RESULT_BASE  = 'ASV_result_random_forest'

os.makedirs(RESULT_BASE, exist_ok=True)
data_files = sorted(glob.glob(DATA_PATTERN))

for file_path in data_files:
    
    base       = os.path.splitext(os.path.basename(file_path))[0]
    level      = base.split('_')[-1]
    result_dir = os.path.join(RESULT_BASE, level)
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n woorking on {level} ===")
    
    ###33 split
    df = pd.read_csv(file_path)
    if 'SampleID' in df.columns:
        df = df.drop(columns=['SampleID'])
    X = df.drop(columns=['Overall_RFI'])
    y = df['Overall_RFI']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    ###grid
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators':     [100, 300, 500, 800, 1200],
        'max_depth':        [None, 10, 20, 30, 50],
        'min_samples_split':[2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features':     ['auto', 'sqrt', 0.3, 0.5]
    }
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    #fitting
    grid.fit(X_train, y_train)
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df.to_csv(os.path.join(result_dir, 'cv_results.csv'), index=False)
    
    best_params = grid.best_params_
    #bestt orameters
    with open(os.path.join(result_dir, 'best_params.txt'), 'w') as f:
        f.write("Best RandomForest Parameters\n")
        f.write("-----------------------------\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
    
    #oob refit
    rf_oob = RandomForestRegressor(**best_params, oob_score=True, random_state=42)
    rf_oob.fit(X_train, y_train)
    
    #####3 oob met
    oob_r2   = rf_oob.oob_score_
    oob_rmse = rmse(y_train, rf_oob.oob_prediction_)
    with open(os.path.join(result_dir, 'oob_metrics.txt'), 'w') as f:
        f.write(f"OOB R2:   {oob_r2:.4f}\n")
        f.write(f"OOB RMSE: {oob_rmse:.4f}\n")
    
    ###### pred met
    y_train_pred = rf_oob.predict(X_train)
    y_test_pred  = rf_oob.predict(X_test)
    metrics = {
        'R2_train':  r2_score(y_train, y_train_pred),
        'RMSE_train':rmse(y_train, y_train_pred),
        'MAE_train': mean_absolute_error(y_train, y_train_pred),
        'R2_test':   r2_score(y_test, y_test_pred),
        'RMSE_test': rmse(y_test, y_test_pred),
        'MAE_test':  mean_absolute_error(y_test, y_test_pred),
    }
    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        f.write("Model Performance Metrics\n")
        f.write("-------------------------\n")
        for name, val in metrics.items():
            f.write(f"{name}: {val:.4f}\n")
    
    ###model file just in case if we decide if this is the best model
    joblib.dump(rf_oob, os.path.join(result_dir, 'best_model.joblib'))
    
    ###fet imp, using everything so many will have zeros
    imp_df = pd.DataFrame({
        'feature':    X.columns,
        'importance': rf_oob.feature_importances_
    }).sort_values('importance', ascending=False)
    imp_df.to_csv(os.path.join(result_dir, 'feature_importances.csv'), index=False)
    
    plt.figure(figsize=(8,6))
    imp_df.head(20).plot.barh(x='feature', y='importance', legend=False)
    plt.gca().invert_yaxis()
    plt.title(f'Top 20 Feature Importances — {level}')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'feature_importances.png'))
    plt.close()
    
    #  t vs t scatter
    plt.figure(figsize=(8,6))
    plt.scatter(y_train, y_train_pred, label='Train', alpha=0.6)
    plt.scatter(y_test,  y_test_pred,  label='Test',  alpha=0.6)
    mn, mx = np.min([y.min(), y_train_pred.min(), y_test_pred.min()]), np.max([y.max(), y_train_pred.max(), y_test_pred.max()])
    plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
    plt.xlabel('Actual Overall_RFI')
    plt.ylabel('Predicted Overall_RFI')
    plt.title(f'Actual vs Predicted (Train & Test) — {level}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'train_test_scatter.png'))
    plt.close()
    
    # tru vs predicted
    idx = np.argsort(y_test.values)
    plt.figure(figsize=(8,6))
    plt.plot(np.array(y_test)[idx],  marker='o', label='Actual')
    plt.plot(y_test_pred[idx],       marker='x', label='Predicted')
    plt.xlabel('Test Samples (sorted by True RFI)')
    plt.ylabel('Overall_RFI')
    plt.title(f'Actual vs Predicted — Test — {level}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    #residuals from test
    residuals = y_test - y_test_pred
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=30)
    plt.title('Residuals Distribution — Test')
    plt.xlabel('Residual')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'residuals_hist.png'))
    plt.close()
    
    plt.figure(figsize=(6,4))
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title('Residuals vs Predicted — Test')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'residuals_vs_pred.png'))
    plt.close()
    
    #Rmse learning 
    train_sizes, train_scores, val_scores = learning_curve(
        rf_oob, X_train, y_train,
        cv=5,
        scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse   = np.sqrt(-val_scores.mean(axis=1))
    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, train_rmse, 'o-', label='Train RMSE')
    plt.plot(train_sizes, val_rmse,   'o-', label='Val   RMSE')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title('Learning Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'learning_curve.png'))
    plt.close()
    
    ###shap
    explainer  = shap.TreeExplainer(rf_oob)
    shap_values = explainer.shap_values(X_test)
    
    ###same in plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(result_dir, 'shap_summary.png'), bbox_inches='tight')
    plt.close()
    
    ###3 tops for extremes 
    abs_resid = np.abs(residuals)
    top_idx   = abs_resid.nlargest(5).index
    records   = []
    for idx in top_idx:
        sv = shap_values[top_idx.get_indexer([idx])[0]]
        feat_idx = np.argsort(np.abs(sv))[-3:][::-1]
        for rank, fi in enumerate(feat_idx, 1):
            records.append({
                'sample':         idx,
                'rank':           rank,
                'feature':        X_test.columns[fi],
                'shap_value':     sv[fi]
            })
    shap_df = pd.DataFrame(records)
    shap_df.to_csv(os.path.join(result_dir, 'extreme_sample_shap.csv'), index=False)
    
    print(f"→ Done {level}: results in `{result_dir}`")

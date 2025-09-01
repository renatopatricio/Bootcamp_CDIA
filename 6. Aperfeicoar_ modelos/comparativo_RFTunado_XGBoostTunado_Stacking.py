import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# 1. Carregar dataset
df = pd.read_csv("D:/bootcamp_cdia/projeto_final/bootcamp_train.csv")
X = df[['temperatura_ar','temperatura_processo','umidade_relativa',
        'velocidade_rotacional','torque','desgaste_da_ferramenta']]
y = df['falha_maquina']

# 2. Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

### ======= Passo 1: Tuning do Random Forest =======
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_dist = {
    'n_estimators': [200, 500],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
rf_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                               n_iter=5, scoring='f1', cv=3, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

### ======= Passo 2: Tuning XGBoost =======
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
param_xgb = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'scale_pos_weight': [scale_pos_weight]
}
xgb_search = RandomizedSearchCV(xgb, param_distributions=param_xgb,
                                n_iter=5, scoring='f1', cv=3, random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

### ======= Passo 3: Ensemble (Stacking) =======
estimators = [
    ('rf', best_rf),
    ('xgb', best_xgb)
]
stacking = StackingClassifier(estimators=estimators,
                              final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
                              n_jobs=-1)
stacking.fit(X_train, y_train)

# Avaliação dos modelos
results = {
    "RandomForest_Tuned": classification_report(y_test, best_rf.predict(X_test), output_dict=True),
    "XGBoost_Tuned": classification_report(y_test, best_xgb.predict(X_test), output_dict=True),
    "Stacking_Ensemble": classification_report(y_test, stacking.predict(X_test), output_dict=True)
}

print(results)
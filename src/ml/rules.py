"""Derive a small, business-readable surrogate for model risk bands."""
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text
from src.data.preprocessor import risk_band

RULE_FEATURES = [
    'EXT_SOURCE_MEAN',
    'CREDIT_INCOME_RATIO',
    'ANNUITY_INCOME_RATIO',
    'AGE_YEARS',
    'EMPLOYED_YEARS',
]


def derive_rules(train_features, train_probability, validation_features,
                 validation_probability, boundaries):
    imputer = SimpleImputer(strategy='median')
    fitted_train = imputer.fit_transform(train_features[RULE_FEATURES])
    model = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=1500, random_state=42
    ).fit(fitted_train, risk_band(train_probability, boundaries))
    predicted = model.predict(imputer.transform(validation_features[RULE_FEATURES]))
    agreement = accuracy_score(
        risk_band(validation_probability, boundaries), predicted
    )
    report = {
        'text': export_text(model, feature_names=RULE_FEATURES),
        'validation_agreement': float(agreement),
        'note': 'Approximation of model risk bands; not lending policy or causal rules.',
    }
    return model, imputer, RULE_FEATURES, report

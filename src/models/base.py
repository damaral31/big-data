# ML models for length of stay prediction
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import pickle
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")


class BaseModel:
    """Base class for all models"""

    def __init__(self, model_name: str, task_type: str = 'regression'):
        self.model_name = model_name
        self.task_type = task_type  # 'regression' or 'classification'
        self.model = None
        self.scaler = StandardScaler()
        self.training_time = 0
        self.feature_names = []
        self.feature_importance = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train model"""
        start_time = time.time()

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled = None

        # Train
        if self.task_type == 'regression':
            self._fit_regression(X_train_scaled, y_train, X_val_scaled, y_val)
        else:
            self._fit_classification(X_train_scaled, y_train, X_val_scaled, y_val)

        self.training_time = time.time() - start_time
        logger.info(f"{self.model_name} trained in {self.training_time:.2f}s")

    def _fit_regression(self, X_train, y_train, X_val=None, y_val=None):
        """Override in subclasses"""
        raise NotImplementedError

    def _fit_classification(self, X_train, y_train, X_val=None, y_val=None):
        """Override in subclasses"""
        raise NotImplementedError

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)"""
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_name} does not support predict_proba")
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importances"""
        if not hasattr(self.model, 'feature_importances_'):
            return None

        importances = self.model.feature_importances_
        if len(self.feature_names) > 0:
            return dict(zip(self.feature_names, importances))
        return None

    def save(self, path: str):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model


class LinearRegressionModel(BaseModel):
    """Linear regression for LOS prediction"""

    def __init__(self):
        super().__init__("Linear Regression", task_type='regression')
        self.model = LinearRegression()

    def _fit_regression(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)


class LogisticRegressionModel(BaseModel):
    """Logistic regression for LOS classification"""

    def __init__(self, max_iter: int = 1000):
        super().__init__("Logistic Regression", task_type='classification')
        self.model = LogisticRegression(max_iter=max_iter, random_state=42)

    def _fit_classification(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)


class RandomForestModel(BaseModel):
    """Random Forest for LOS prediction"""

    def __init__(self, n_estimators: int = 100, task_type: str = 'regression',
                 max_depth: int = 20, min_samples_split: int = 5):
        super().__init__("Random Forest", task_type=task_type)
        if task_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )

    def _fit_regression(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

    def _fit_classification(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting for LOS prediction"""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 5, task_type: str = 'regression'):
        super().__init__("Gradient Boosting", task_type=task_type)
        if task_type == 'regression':
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )

    def _fit_regression(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(self.scaler.fit_transform(X_val), y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def _fit_classification(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)


class XGBoostModel(BaseModel):
    """XGBoost model for LOS prediction"""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6, task_type: str = 'regression'):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")

        super().__init__("XGBoost", task_type=task_type)

        if task_type == 'regression':
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

    def _fit_regression(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(self.scaler.fit_transform(X_val), y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def _fit_classification(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)


class LightGBMModel(BaseModel):
    """LightGBM model for LOS prediction"""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 8, task_type: str = 'regression'):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")

        super().__init__("LightGBM", task_type=task_type)

        if task_type == 'regression':
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

    def _fit_regression(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(self.scaler.fit_transform(X_val), y_val)],
                          verbose=-1)
        else:
            self.model.fit(X_train, y_train)

    def _fit_classification(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)


class SVRModel(BaseModel):
    """Support Vector Regression for LOS prediction"""

    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        super().__init__("SVR", task_type='regression')
        self.model = SVR(kernel=kernel, C=C)

    def _fit_regression(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)


class KNNModel(BaseModel):
    """K-Nearest Neighbors for LOS prediction"""

    def __init__(self, n_neighbors: int = 5):
        super().__init__("KNN", task_type='regression')
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)

    def _fit_regression(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)


class ModelFactory:
    """Factory for creating and managing models"""

    REGRESSION_MODELS = {
        'linear': LinearRegressionModel,
        'rf': RandomForestModel,
        'gb': GradientBoostingModel,
        'svr': SVRModel,
        'knn': KNNModel,
    }

    CLASSIFICATION_MODELS = {
        'logistic': LogisticRegressionModel,
        'rf': lambda: RandomForestModel(task_type='classification'),
    }

    @staticmethod
    def get_available_models(task_type: str = 'regression') -> List[str]:
        """Get list of available models"""
        if task_type == 'regression':
            models = list(ModelFactory.REGRESSION_MODELS.keys())
        else:
            models = list(ModelFactory.CLASSIFICATION_MODELS.keys())

        # Add conditional models
        if XGBOOST_AVAILABLE:
            models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            models.append('lightgbm')

        return models

    @staticmethod
    def create_model(model_name: str, task_type: str = 'regression', **kwargs) -> BaseModel:
        """Create a model instance"""
        if task_type == 'regression':
            if model_name in ModelFactory.REGRESSION_MODELS:
                return ModelFactory.REGRESSION_MODELS[model_name](**kwargs)
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                return XGBoostModel(**kwargs)
            elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                return LightGBMModel(**kwargs)
        else:
            if model_name in ModelFactory.CLASSIFICATION_MODELS:
                return ModelFactory.CLASSIFICATION_MODELS[model_name](**kwargs)

        raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def create_all_models(task_type: str = 'regression') -> List[BaseModel]:
        """Create all available models"""
        available_models = ModelFactory.get_available_models(task_type)
        models = []

        for model_name in available_models:
            try:
                model = ModelFactory.create_model(model_name, task_type)
                models.append(model)
            except Exception as e:
                logger.warning(f"Failed to create {model_name}: {e}")

        return models

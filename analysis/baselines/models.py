"""
Baseline models for metabarcoding relative abundance prediction.
Includes simple baselines, linear models, tree-based models, and zero-inflated models.
"""
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor


class BaselineModel(ABC):
    """Abstract base class for all baseline models."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.fitted = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaselineModel":
        """Fit the model on training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# Simple Baselines
# =============================================================================

class MeanBaseline(BaselineModel):
    """Predicts the mean of training targets."""
    
    def __init__(self):
        super().__init__(name="Mean Baseline")
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MeanBaseline":
        self.mean_value = y.mean()
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_value)


class ZeroBaseline(BaselineModel):
    """Always predicts zero (useful for highly zero-inflated data)."""
    
    def __init__(self):
        super().__init__(name="Zero Baseline")
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ZeroBaseline":
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


# =============================================================================
# Linear Models
# =============================================================================

class LinearRegressionModel(BaselineModel):
    """Standard linear regression."""
    
    def __init__(self):
        super().__init__(name="Linear Regression")
        self.model = LinearRegression()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LinearRegressionModel":
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, 1)  # Clip to valid probability range


class RidgeRegressionModel(BaselineModel):
    """Ridge regression with L2 regularization."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__(name=f"Ridge Regression (alpha={alpha})")
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "RidgeRegressionModel":
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, 1)


class ElasticNetModel(BaselineModel):
    """Elastic Net with L1 + L2 regularization."""
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5):
        super().__init__(name=f"ElasticNet (alpha={alpha}, l1={l1_ratio})")
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ElasticNetModel":
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, 1)


# =============================================================================
# Tree-Based Models
# =============================================================================

class DecisionTreeModel(BaselineModel):
    """Decision Tree regressor."""
    
    def __init__(self, max_depth: int = 10, min_samples_leaf: int = 5):
        super().__init__(name=f"Decision Tree (depth={max_depth})")
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "DecisionTreeModel":
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, 1)


class RandomForestModel(BaselineModel):
    """Random Forest regressor."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 15, 
                 min_samples_leaf: int = 5):
        super().__init__(name=f"Random Forest (n={n_estimators}, depth={max_depth})")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "RandomForestModel":
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, 1)


class GradientBoostingModel(BaselineModel):
    """Gradient Boosting regressor."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, 
                 learning_rate: float = 0.1):
        super().__init__(name=f"Gradient Boosting (n={n_estimators}, lr={learning_rate})")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "GradientBoostingModel":
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, 1)


# =============================================================================
# Other Standard Models
# =============================================================================

class KNNModel(BaselineModel):
    """K-Nearest Neighbors regressor."""
    
    def __init__(self, n_neighbors: int = 78):
        super().__init__(name=f"KNN (k={n_neighbors})")
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "KNNModel":
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, 1)


# =============================================================================
# Zero-Inflated Models (for handling excess zeros)
# =============================================================================

class TwoStageModel(BaselineModel):
    """
    Two-stage model for zero-inflated data:
    1. Classification stage: Predict presence/absence (binary)
    2. Regression stage: Predict abundance given presence
    
    Final prediction = P(presence) * E[abundance | presence]
    """
    
    def __init__(self, classifier=None, regressor=None, threshold: float = 0.5):
        super().__init__(name="Two-Stage (Hurdle) Model")
        # Tuned hyperparameters from random_forest folder
        self.classifier = classifier or RandomForestClassifier(
            n_estimators=200, 
            max_depth=20, 
            min_samples_split=10,
            class_weight='balanced_subsample',
            random_state=42, 
            n_jobs=-1
        )
        self.regressor = regressor or RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=10,
            min_samples_leaf=1,
            random_state=42, 
            n_jobs=-1
        )
        self.threshold = threshold
        
    def fit(self, X: pd.DataFrame, y: pd.Series, presence: pd.Series = None, 
            **kwargs) -> "TwoStageModel":
        """
        Fit both stages.
        
        Args:
            X: Features
            y: Target (relative abundance)
            presence: Binary presence indicator (if None, derived from y > 0)
        """
        if presence is None:
            presence = (y > 0).astype(int)
        
        # Stage 1: Fit classifier on all data
        self.classifier.fit(X, presence)
        
        # Stage 2: Fit regressor only on positive samples
        positive_mask = presence > 0
        if positive_mask.sum() > 0:
            X_positive = X.loc[positive_mask]
            y_positive = y.loc[positive_mask]
            self.regressor.fit(X_positive, y_positive)
        
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Get probability of presence
        prob_presence = self.classifier.predict_proba(X)[:, 1]
        
        # Get predicted abundance (for all, but will weight by presence prob)
        pred_abundance = self.regressor.predict(X)
        
        # Final prediction: P(presence) * E[abundance | presence]
        predictions = prob_presence * pred_abundance
        return np.clip(predictions, 0, 1)
    
    def predict_presence(self, X: pd.DataFrame) -> np.ndarray:
        """Get just the presence probabilities."""
        return self.classifier.predict_proba(X)[:, 1]


class ZeroInflatedRidge(BaselineModel):
    """
    Zero-inflated regression using Ridge:
    - Uses a modified target: log(y + epsilon) for non-zero values
    - Explicitly models zeros with a classification component
    """
    
    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-6):
        super().__init__(name=f"Zero-Inflated Ridge (alpha={alpha})")
        self.alpha = alpha
        self.epsilon = epsilon
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.regressor = Ridge(alpha=alpha)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ZeroInflatedRidge":
        # Binary classification for presence
        presence = (y > 0).astype(int)
        self.classifier.fit(X, presence)
        
        # Regression on log-transformed positive values
        positive_mask = y > 0
        if positive_mask.sum() > 0:
            X_pos = X.loc[positive_mask]
            y_pos = np.log(y.loc[positive_mask] + self.epsilon)
            self.regressor.fit(X_pos, y_pos)
        
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob_presence = self.classifier.predict_proba(X)[:, 1]
        log_abundance = self.regressor.predict(X)
        abundance = np.exp(log_abundance) - self.epsilon
        
        predictions = prob_presence * np.clip(abundance, 0, 1)
        return np.clip(predictions, 0, 1)


class TweedieModel(BaselineModel):
    """
    Tweedie regression for zero-inflated continuous data.
    Uses sklearn's TweedieRegressor (requires sklearn >= 1.0)
    """
    
    def __init__(self, power: float = 1.5, alpha: float = 1.0):
        super().__init__(name=f"Tweedie (power={power})")
        self.power = power
        self.alpha = alpha
        try:
            from sklearn.linear_model import TweedieRegressor
            self.model = TweedieRegressor(power=power, alpha=alpha, max_iter=5000)
            self.available = True
        except ImportError:
            self.available = False
            self.model = Ridge(alpha=alpha)
            print("Warning: TweedieRegressor not available, falling back to Ridge")
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "TweedieModel":
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, 1)


class LogTransformModel(BaselineModel):
    """
    Model that applies log transformation to handle zero-inflation:
    y_transformed = log(y + epsilon)
    Uses any base regressor on transformed targets.
    """
    
    def __init__(self, base_model=None, epsilon: float = 1e-6):
        super().__init__(name="Log-Transform Model")
        self.epsilon = epsilon
        self.base_model = base_model or RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LogTransformModel":
        # Transform target
        y_transformed = np.log(y + self.epsilon)
        self.base_model.fit(X, y_transformed)
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_preds = self.base_model.predict(X)
        preds = np.exp(log_preds) - self.epsilon
        return np.clip(preds, 0, 1)


class QuantileRandomForest(BaselineModel):
    """
    Quantile regression with Random Forest.
    Useful for zero-inflated data as it can learn asymmetric distributions.
    Uses median prediction (quantile=0.5) by default.
    """
    
    def __init__(self, n_estimators: int = 100, quantile: float = 0.5):
        super().__init__(name=f"Quantile RF (q={quantile})")
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "QuantileRandomForest":
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Get predictions from all trees
        all_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        # Compute quantile prediction
        preds = np.percentile(all_preds, self.quantile * 100, axis=0)
        return np.clip(preds, 0, 1)


# =============================================================================
# Latent + MLP Model (from src/)
# =============================================================================

class LatentMLPModel(BaselineModel):
    """
    Wrapper around the Latent+MLP Trainer from src/.
    Uses cross-entropy loss (softmax per sample).
    Requires fixed_split_indices matching the baselines' random split to ensure
    the same test set is used for fair comparison.
    """

    def __init__(self, fixed_split_indices: Optional[Dict] = None):
        super().__init__(name="Latent MLP")
        self.fixed_split_indices = fixed_split_indices
        # (sample_id, bin_uri) -> predicted relative abundance
        self._pred_lookup: Dict[tuple, float] = {}
        self.fitted = False

    def fit(self, X, y, **kwargs) -> "LatentMLPModel":
        """Train the Latent+MLP model using the Trainer from src/."""
        import sys
        src_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')
        )
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from config import Config, set_seed
        from train import Trainer

        set_seed(42)
        cfg = Config()
        trainer = Trainer(
            cfg,
            fixed_split_indices=self.fixed_split_indices,
        )
        trainer.run(use_wandb=False)

        # Build (sample_id, bin_uri) -> prediction lookup from flat labeled arrays.
        preds_flat, _, sample_labels, bin_labels = trainer.get_predictions("test")
        self._pred_lookup = {
            (str(sample_labels[i]), str(bin_labels[i])): float(preds_flat[i])
            for i in range(len(preds_flat))
        }

        self.fitted = True
        return self

    def predict(self, X, test_meta=None) -> np.ndarray:
        """
        Return predictions aligned to the baseline test set ordering.
        test_meta must be a DataFrame with columns 'sample_id' and 'bin_uri'
        (from metadata['test_meta'] in run_visualization).
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        if test_meta is None:
            return np.array(list(self._pred_lookup.values()), dtype=np.float64)
        return np.array(
            [self._pred_lookup.get((row.sample_id, row.bin_uri), 0.0)
             for row in test_meta.itertuples()],
            dtype=np.float64,
        )


# =============================================================================
# Model Registry
# =============================================================================

def get_all_models() -> Dict[str, BaselineModel]:
    """Get a dictionary of all available baseline models."""
    return {
        # Simple baselines
        "mean": MeanBaseline(),
        "zero": ZeroBaseline(),
        
        # Linear models
        "linear_regression": LinearRegressionModel(),
        "ridge": RidgeRegressionModel(alpha=1.0),
        "ridge_high_reg": RidgeRegressionModel(alpha=10.0),
        "elasticnet": ElasticNetModel(alpha=0.1, l1_ratio=0.5),
        
        # Tree-based models
        "decision_tree": DecisionTreeModel(max_depth=10),
        "random_forest": RandomForestModel(n_estimators=100, max_depth=15),
        "gradient_boosting": GradientBoostingModel(n_estimators=100),
        
        # Other models
        "knn": KNNModel(n_neighbors=78),
        
        # Zero-inflated models
        "two_stage": TwoStageModel(),
        "zero_inflated_ridge": ZeroInflatedRidge(alpha=1.0),
        "tweedie": TweedieModel(power=1.5),
        "log_transform": LogTransformModel(),
        "quantile_rf": QuantileRandomForest(quantile=0.5),
    }


def get_zero_inflated_models() -> Dict[str, BaselineModel]:
    """Get only zero-inflated models."""
    return {
        "two_stage": TwoStageModel(),
        "zero_inflated_ridge": ZeroInflatedRidge(alpha=1.0),
        "tweedie": TweedieModel(power=1.5),
        "log_transform": LogTransformModel(),
        "quantile_rf": QuantileRandomForest(quantile=0.5),
    }

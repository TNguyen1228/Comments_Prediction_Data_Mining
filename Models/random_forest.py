
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeRegressor
from scipy import stats

class RandomForestRegressor():
    """Implement Random Forest regressor from scratch using Decision Tree."""
    
    def __init__(
        self,
        n_estimators=100,
        criterion='mse', 
        max_depth=None,
        min_samples_leaf=1,
        max_features='sqrt', 
        min_impurity_decrease=0.0,
        random_state=0
    ):
        """
        Some important parameters in Random Forest.
        
        Args:
            n_estimators (int): The number of trees in the forest.
            criterion (str): The function to measure the quality of a split ('mse' for mean squared error).
            max_depth (int): The maximum depth of the tree.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            max_features (str): The number of features to consider when looking for the best split; 'sqrt' for square root.
            min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            random_state (int): Controls randomness of the bootstrap samples and the features.
        """
        self.n_estimators = n_estimators
        self.criterion =  criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        
    def fit(self, X, y):
        """Fit the random forest model."""
        self.n_samples, self.n_features = X.shape
        if self.max_features == 'sqrt':
            self.max_feature = int(np.sqrt(self.n_features))
        
        self.trees = []
        for i in range(self.n_estimators):
            X_train, _, y_train, _ = train_test_split(
                X, 
                y, 
                test_size=0.3, 
                random_state=self.random_state + i
            )
            tree = DecisionTreeRegressor(
                criterion = self.criterion,
                max_depth = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                max_features = self.max_features,
                random_state = self.random_state
            )
            tree.fit(X_train, y_train)
            self.trees.append(tree)
    
    def predict(self, X_test):
        """Predict continuous values for X_test."""
        predictions = np.array([tree.predict(X_test) for tree in self.trees])
        predicted_values = np.mean(predictions, axis=0)
        return predicted_values

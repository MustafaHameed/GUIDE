from sklearn.linear_model import LogisticRegression


def create_model():
    """Create and configure the logistic regression model."""
    return LogisticRegression(max_iter=1000)

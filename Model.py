# http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
import numpy as np
from scipy.optimize import fmin_tnc


class LogisticRegressionUsingGD:

    def __init__(self, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon
        self.perturbed_x = None

    @staticmethod
    def sigmoid(x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def net_input(theta, x):
        return np.dot(x, theta)

    def probability(self, theta, x):
        # Calculates the probability that an instance belongs to a particular class

        return self.sigmoid(self.net_input(theta, x))

    def entropy_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            y * np.log(self.probability(theta, x)) + (1 - y) * np.log(1 - self.probability(theta, x)))
        return total_cost

    def entropy_gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        total_gradient = (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)
        return total_gradient

    def ridge_function(self, theta, x, y):
        total_cost = self.entropy_function(theta, x, y) + .5 * self.alpha * theta.dot(theta)
        return total_cost

    def ridge_gradient(self, theta, x, y):
        total_gradient = self.entropy_gradient(theta, x, y) + self.alpha * theta
        return total_gradient

    def fgsm_function(self, theta, x, y):
        signed_grad = np.sign(self.entropy_gradient(theta, x, y))
        self.perturbed_x = x + (self.epsilon * signed_grad)
        total_cost = (self.alpha * self.entropy_function(theta, x, y)) + ((1 - self.alpha) * self.entropy_function(theta, self.perturbed_x, y))
        return total_cost

    def fsgm_gradient(self, theta, x, y):
        total_gradient = (self.alpha * self.entropy_gradient(theta, x, y)) + ((1 - self.alpha) * self.entropy_gradient(
            theta, self.perturbed_x, y))
        return total_gradient

    def fit(self, x, y, theta, type_penalitty=None):
        """trains the model from the training data
        Uses the fmin_tnc function that is used to find the minimum for any function
        It takes arguments as
            1) func : function to minimize
            2) x0 : initial values for the parameters
            3) fprime: gradient for the function defined by 'func'
            4) args: arguments passed to the function
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            Training samples
        y: array-like, shape = [n_samples, n_target_values]
            Target classes
        theta: initial weights
        type_penalitty : type of penalitty
        Returns
        -------
        self: An instance of self
        """

        if type_penalitty == 'RIDGE':
            self.cost_function = self.ridge_function
            self.gradient = self.ridge_gradient
        elif type_penalitty == 'FSGM':
            self.cost_function = self.fgsm_function
            self.gradient = self.fsgm_gradient
        else:
            self.cost_function = self.entropy_function
            self.gradient = self.entropy_gradient

        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient,
                               args=(x, y.flatten()))
        self.w_ = opt_weights[0]
        return self

    def predict(self, x):
        """ Predicts the class labels
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        predicted class labels
        """
        theta = self.w_[:, np.newaxis]
        return self.probability(theta, x)

    def accuracy(self, x, actual_classes, probab_threshold=0.5):
        """Computes the accuracy of the classifier
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            Training samples
        actual_classes : class labels from the training data set
        probab_threshold: threshold/cutoff to categorize the samples into different classes
        Returns
        -------
        accuracy: accuracy of the model
        """
        predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100

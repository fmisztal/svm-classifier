from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from data_processing import *

class SVM:
    def __init__(self, learning_rate, epochs, regularization_param, kernel=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_param = regularization_param
        self.kernel = kernel
        self.kernel_function = None
        self.weights = None
        self.bias = None
        self._set_kernel_function()

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            decision_function = np.array([self.kernel_function(x, self.weights) for x in X]) + self.bias
            hinge_loss = np.maximum(0, 1 - y * decision_function)
            loss = np.sum(hinge_loss) + 0.5 * self.regularization_param * np.dot(self.weights, self.weights)
            gradient_weights = -np.dot(X.T, y * (decision_function < 1)) + self.regularization_param * self.weights
            gradient_bias = -np.sum(y * (decision_function < 1))
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss}")

    def predict(self, X):
        decision_function = np.dot(X, self.weights) + self.bias
        predictions = np.sign(decision_function)
        return predictions

    def _set_kernel_function(self):
        if self.kernel == 'linear':
            self.kernel_function = self.linear_kernel
        elif self.kernel == 'poly':
            self.kernel_function = self.polynomial_kernel
        elif self.kernel == 'rbf':
            self.kernel_function = self.gaussian_kernel
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def linear_kernel(self, x1, x2):
        return np.dot(x1.T, x2)

    def polynomial_kernel(self, x1, x2, degree=2, coef0=1):
        return (np.dot(x1.T, x2) + coef0) ** degree

    def gaussian_kernel(self, x1, x2, sigma=0.1):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))


if __name__ == '__main__':
    # X_train, X_test, y_train, y_test = load_quality_mean_data()
    # X_train, X_test, y_train, y_test = load_quality_data()
    X_train, X_test, y_train, y_test = load_color_data()

    svm = SVM(learning_rate=1, epochs=10, regularization_param=0.01, kernel='linear')
    svm.train(X_train, y_train)
    y_predict = svm.predict(X_test)


    conf_matrix = confusion_matrix(y_test, y_predict)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
    print(f"Accuracy: {accuracy}")

    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    print(f"Sensitivity: {sensitivity}")

    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print(f"Specificity: {specificity}")

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')

    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xticks([0, 1], ['Predicted -1', 'Predicted 1'])
    plt.yticks([0, 1], ['True -1', 'True 1'])
    plt.show()


    df = pd.DataFrame({'y_predict': y_predict})
    df.to_csv('y_predict.csv', index=False)
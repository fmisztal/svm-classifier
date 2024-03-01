from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from data_processing import *
from svm import *


def test_learning_rate():
    learning_rates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    epochs = [10, 100, 1000, 10000]

    X_train, X_test, y_train, y_test = load_quality_data()
    for ep in epochs:
        accuracies = []
        for lr in learning_rates:
            svm = SVM(learning_rate=lr, epochs=ep, regularization_param=0.01, kernel='linear')
            svm.train(X_train, y_train)
            y_predict = svm.predict(X_test)

            conf_matrix = confusion_matrix(y_test, y_predict)
            accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
            accuracies.append(accuracy)

            print(ep, '  ', lr)

        plt.plot(learning_rates, accuracies, marker='o', linestyle='-', label=f'{ep} epochs')

    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Accuracy')
    plt.title(f'Relationship between learning rate and accuracy')
    plt.legend()
    plt.show()

def test_regulation_param():
    learning_rates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    regulation_params = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    X_train, X_test, y_train, y_test = load_quality_data()
    for rp in regulation_params:
        accuracies = []
        for lr in learning_rates:
            svm = SVM(learning_rate=lr, epochs=1000, regularization_param=rp, kernel='linear')
            svm.train(X_train, y_train)
            y_predict = svm.predict(X_test)

            conf_matrix = confusion_matrix(y_test, y_predict)
            accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
            accuracies.append(accuracy)

            print(rp, '  ', lr)

        plt.plot(learning_rates, accuracies, marker='o', linestyle='-', label=f'regulation {rp} ')

    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Accuracy')
    plt.title(f'Relationship between learning rate and accuracy')
    plt.legend()
    plt.show()

def test_regulation_param_2():
    epochs = [10, 100, 1000, 10000]
    regulation_params = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    X_train, X_test, y_train, y_test = load_quality_data()
    for ep in epochs:
        accuracies = []
        for rp in regulation_params:
            svm = SVM(learning_rate=0.001, epochs=ep, regularization_param=rp, kernel='linear')
            svm.train(X_train, y_train)
            y_predict = svm.predict(X_test)

            conf_matrix = confusion_matrix(y_test, y_predict)
            accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
            accuracies.append(accuracy)

            print(rp, '  ', ep)

        plt.plot(regulation_params, accuracies, marker='o', linestyle='-', label=f'{ep} epochs')

    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Accuracy')
    plt.title(f'Relationship between learning rate and accuracy')
    plt.legend()
    plt.show()

def test_kernels():
    kernels = ['linear', 'rbf']

    for i in range(3):
        X_train, X_test, y_train, y_test = None, None, None, None
        if i == 0:
            X_train, X_test, y_train, y_test = load_quality_data()
        elif i == 1:
            X_train, X_test, y_train, y_test = load_quality_mean_data()
        elif i == 2:
            X_train, X_test, y_train, y_test = load_color_data()

        for kernel in kernels:
            svm = SVM(learning_rate=0.00001, epochs=1000, regularization_param=0.00001, kernel=kernel)
            svm.train(X_train, y_train)
            y_predict = svm.predict(X_test)

            conf_matrix = confusion_matrix(y_test, y_predict)
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

            print(kernel)
            accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
            print(f"Accuracy: {(accuracy*100):.2f}%")

            sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
            print(f"Sensitivity: {(sensitivity*100):.2f}%")

            specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
            print(f"Specificity: {(specificity*100):.2f}%\n")

            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')

            plt.colorbar()
            plt.title(f'Confusion Matrix for {kernel} kernel')
            plt.xticks([0, 1], ['Predicted -1', 'Predicted 1'])
            plt.yticks([0, 1], ['True -1', 'True 1'])
            plt.show()
        print(f'\n====================\n')


if __name__ == '__main__':
    test_learning_rate()
    test_regulation_param()
    test_regulation_param_2()
    test_kernels()
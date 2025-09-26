import time
import numpy as np
import matplotlib.pyplot as plt
from utils.feature_normalization_util import z_score_normalize
from utils.gradient_descent_util import run_gradient_descent, predict

#np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
# y = 1 + x**2, no engineering
def no_engineering():
    x = np.arange(0, 20, 1)
    y = 1 + x**2

    X = x.reshape(-1, 1)
    iterations = 900
    w_final, b_final, J_hist = run_gradient_descent(X, y, iterations, 1e-2)
    #print(w_final, b_final)
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].scatter(x, y, marker = 'x', c='r', label = "Actual Value")
    ax[0].plot(x, X@w_final + b_final, label = "Predicted Value")
    ax[0].set_title("Model"); ax[0].set_xlabel('x'); ax[0].set_ylabel('y'); ax[0].legend()
    x_iter = np.arange(0, iterations, 1)
    y_loss = np.zeros((iterations,))
    for i in range(iterations):
        y_loss[i] = J_hist[i][0]
    ax[1].plot(x_iter, y_loss)
    ax[1].set_title("Loss vs Iterations"); ax[1].set_xlabel('Iterations'); ax[1].set_ylabel('Loss')
    fig.suptitle("Simple quadratic func, no feature engineering")
    plt.show()
    
# y = 1 + x**2, -> model: y = w * x**2 + b
def polynomal_feature():
    x = np.arange(0, 20, 1)
    y = 1 + x**2

    # engineer features .
    X = x**2 #<-- added engineered feature

    X = X.reshape(-1, 1)
    iterations = 10000
    w_final, b_final, J_hist = run_gradient_descent(X, y, iterations, 1e-5)

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].scatter(x, y, marker = 'x', c='r', label = "Actual Value")
    ax[0].plot(x, X@w_final + b_final, label = "Predicted Value")
    ax[0].set_title("Model, Added x**2 feature")
    ax[0].set_xlabel('x'); ax[0].set_ylabel('y')
    ax[0].legend()
    x_iter = np.arange(0, iterations, 1)
    y_loss = np.zeros((iterations,))
    for i in range(iterations):
        y_loss[i] = J_hist[i][0]
    ax[1].plot(x_iter, y_loss)
    ax[1].set_title("Loss vs Iterations")
    ax[1].set_xlabel('Iterations'); ax[1].set_ylabel('Loss')
    fig.suptitle("Polinomal feature, y = w * x**2 + b")
    plt.show()

# y = x**2, -> model: y = w0*x1 + w1 * x2**2 + w3 * x3**3 + b
def polynomal_feature_selected():
    x = np.arange(0, 20, 1)

    y = x**2

    # engineer features
    X = np.c_[x, x**2, x**3] #<-- added engineered feature

    #X = X.reshape(-1, 1)
    iterations = 10000
    w_final, b_final, J_hist = run_gradient_descent(X, y, iterations, 1e-7)

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].scatter(x, y, marker = 'x', c='r', label = "Actual Value")
    ax[0].plot(x, X@w_final + b_final, label = "Predicted Value")
    ax[0].set_title("Model")
    ax[0].set_xlabel('x'); ax[0].set_ylabel('y')
    ax[0].legend()
    x_iter = np.arange(0, iterations, 1)
    y_loss = np.zeros((iterations,))
    for i in range(iterations):
        y_loss[i] = J_hist[i][0]
    ax[1].plot(x_iter, y_loss)
    ax[1].set_title("Loss vs Iterations")
    ax[1].set_xlabel('Iterations'); ax[1].set_ylabel('Loss')
    fig.suptitle("x, x**2, x**3 features")
    plt.show()
    #less weight value implies less important/correct feature, and in extreme, when the weight becomes zero or 
    # very close to zero, the associated feature is not useful in fitting the model to the data.
    #above, after fitting, the weight associated with the  𝑥2
    #feature is much larger than the weights for  𝑥 or  𝑥3 as it is the most useful in fitting the data.

def plot_polynomal_features():
    x = np.arange(0, 20, 1)
    y = x**2

    # engineer features .
    X = np.c_[x, x**2, x**3] #<-- added engineered feature
    X_features = ['x', 'x^2', 'x^3']

    fig, ax = plt.subplots(1, 3, figsize=(12,3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X[:, i], y)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("Y")
    
    fig.suptitle("Polynomal features")
    plt.show()
    #Another way to think about this is to note that we are still using linear regression 
    # once we have created new features. Given that, the best features will be linear relative to the target.
    #Above, it is clear that the  𝑥2
    #feature mapped against the target value  𝑦
    #is linear. Linear regression can then easily generate a model using that feature.

# y = x**2, -> model: complex
def polinomal_feature_selected_test():
    x = np.arange(0, 20, 1)
    a = np.arange(20)
    a = np.random.rand(20) * 20
    print(a)
    x_ = np.c_[x, a]
    print(x_)

    y = x**2

    # engineer features
    X = np.c_[x_, x_**2, x_**3, x_**4] #<-- added engineered feature
    print(X)
    #X = X.reshape(-1, 1)
    iterations = 10000
    w_final, b_final, J_hist = run_gradient_descent(X, y, iterations, 1e-10)

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].scatter(x, y, marker = 'x', c='r', label = "Actual Value")
    ax[0].plot(x, X@w_final + b_final, label = "Predicted Value")
    ax[0].set_title("Model")
    ax[0].set_xlabel('x'); ax[0].set_ylabel('y')
    ax[0].legend()
    x_iter = np.arange(0, iterations, 1)
    y_loss = np.zeros((iterations,))
    for i in range(iterations):
        y_loss[i] = J_hist[i][0]
    ax[1].plot(x_iter, y_loss)
    ax[1].set_title("Loss vs Iterations")
    ax[1].set_xlabel('Iterations'); ax[1].set_ylabel('Loss')
    fig.suptitle("x, x**2, x**3 features")
    plt.show()
    #less weight value implies less important/correct feature, and in extreme, when the weight becomes zero or 
    # very close to zero, the associated feature is not useful in fitting the model to the data.
    #above, after fitting, the weight associated with the  𝑥2
    #feature is much larger than the weights for  𝑥 or  𝑥3 as it is the most useful in fitting the data.


def plot_polynomal_features():
    x = np.arange(0, 20, 1)
    y = x**2

    # engineer features .
    X = np.c_[x, x**2, x**3] #<-- added engineered feature
    X_features = ['x', 'x^2', 'x^3']

    fig, ax = plt.subplots(1, 3, figsize=(12,3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X[:, i], y)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("Y")
    
    fig.suptitle("Polynomal features")
    plt.show()

def scaling_polynomal_features():
    x = np.arange(0, 20, 1)
    y = x**2

    # engineer features
    X0 = np.c_[x, x**2, x**3] #<-- added engineered feature
    print(f"Peak to Peak range by columb in Raw     X:{np.ptp(X0, axis=0)}")

    X, mean, sigma = z_score_normalize(X0)
    print(f"Peak to Peak range by columb in Normalized     X:{np.ptp(X, axis=0)}")

    #X = X.reshape(-1, 1)
    iterations = 100000
    w_final, b_final, J_hist = run_gradient_descent(X0, y, iterations, 1e-7, 0.01)

    w_final_norm, b_final_norm, J_hist_norm = run_gradient_descent(X, y, iterations, 1e-1, 0.01)

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].scatter(x, y, marker = 'x', c='r', label = "Actual Value")
    ax[0].plot(x, X0@w_final + b_final, label = "Predicted Value Raw")
    ax[0].plot(x, X@w_final_norm + b_final_norm, label = "Predicted Value Normalized")
    ax[0].set_title("Model")
    ax[0].set_xlabel('x'); ax[0].set_ylabel('y')
    ax[0].legend()
    x_raw = np.arange(0, len(J_hist), 1)
    y_loss_raw = np.zeros((len(J_hist),))
    for i in range(len(J_hist)):
            y_loss_raw[i] = J_hist[i][0]
    x_norm = np.arange(0, len(J_hist_norm), 1)
    y_loss_norm = np.zeros((len(J_hist_norm),))
    for i in range(len(J_hist_norm)):
            y_loss_norm[i] = J_hist_norm[i][0]
    ax[1].plot(x_raw, y_loss_raw, label = "Raw")
    ax[1].plot(x_norm, y_loss_norm, label = "Normalized")
    ax[1].set_title("Loss vs Iterations")
    ax[1].set_xlabel('Iterations'); ax[1].set_ylabel('Loss')
    fig.suptitle("Normalized x, x**2, x**3 feature")
    plt.show()

def complex_polynomal_function():
    x = np.arange(0, 20, 1)
    y = np.cos(x/2)

    # engineer features
    X0 = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13] #<-- added engineered feature

    X, mean, sigma = z_score_normalize(X0)

    iterations = 1000000
    w_final, b_final, J_hist = run_gradient_descent(X, y, iterations, 1e-1, threshold=1e-10)

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].scatter(x, y, marker = 'x', c='r', label = "Actual Value")
    ax[0].plot(x, X@w_final + b_final, label = "Predicted Value")
    ax[0].set_title("Model")
    ax[0].set_xlabel('x'); ax[0].set_ylabel('y')
    ax[0].legend()
    x_iter = np.arange(0, iterations, 1)
    y_loss = np.zeros((iterations,))
    for i in range(iterations):
        y_loss[i] = J_hist[i][0]
    ax[1].plot(x_iter, y_loss)
    ax[1].set_title("Loss vs Iterations")
    ax[1].set_xlabel('Iterations'); ax[1].set_ylabel('Loss')
    fig.suptitle("Complex function")
    plt.show()


if __name__ == "__main__":
    #no_engineering()
    #polynomal_feature()
    #polynomal_feature_selected()
    #plot_polynomal_features()
    #polynomal_feature_selected_test()
    #scaling_polynomal_features()
    complex_polynomal_function()

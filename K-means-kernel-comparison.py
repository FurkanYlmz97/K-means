__Author__: "Furkan Yılmaz"

from mlxtend.data import loadlocal_mnist
import numpy as np
import matplotlib.pyplot as plt
import time


def createKernel(X, type=None, c_poly=0, d=1, std=0.1, c_sigmoid=1, theta=0):

    X = X / 255
    # X = X - np.mean(X)

    if type == "Gaussian":
        A = X.T @ X
        K = np.zeros((len(X.T), len(X.T)))
        for m in range(len(X.T)):
            for n in range(len(X.T)):
                K[m, n] = np.exp(- (A[m, m] + A[n, n] - 2*A[m, n]) / (2*(std**2)))

    elif type == "Polynomial":
        K = (X.T @ X + c_poly)**d

    elif type == "Sigmoid":
        K = np.tanh((c_sigmoid*X).T @ X + theta)

    else:
        K = X.T @ X

    return K


def createK2(X, B, type=None, c_poly=0, d=1, std=0.1, c_sigmoid=1, theta=0):

    X = X / 255
    B = B / 255
    # X = X - np.mean(X)

    if type == "Gaussian":
        A = X.T @ B
        K = np.zeros((len(X.T), len(X.T)))
        for m in range(len(X.T)):
            for n in range(len(X.T)):
                K[m, n] = np.exp(- (A[m, m] + A[n, n] - 2*A[m, n]) / (2*(std**2)))

    elif type == "Polynomial":
        K = (X.T @ B + c_poly)**d

    elif type == "Sigmoid":
        K = np.tanh((c_sigmoid*X).T @ B + theta)

    else:
        K = X.T @ B

    return K


def trainKmean(X, max_iter, k, kernel=None, c_poly=0, d=1, std=0.1, c_sigmoid=1, theta=0):
    # start_time = time.time()
    K = createKernel(X, kernel, c_poly, d, std, c_sigmoid, theta)
    # print("--- %s seconds for creating Kernel ---" % (time.time() - start_time))
    I = None
    epoch_error = []
    W = None

    max_try = 5
    m = 0
    restart = True
    while restart:
        restart = False
        epoch_error = []

        I = np.zeros((X.shape[1], k))
        for i in range(len(I)):
            rand = np.random.randint(k)
            I[i, rand] = 1

        W = I * (1 / np.sum(I, axis=0))
        restart = np.isnan(np.sum(W))

        for _ in range(max_iter):

            A = W.T @ K
            B = A @ W

            P = ((-2 * A) + np.diag(K)).T + np.diag(B)
            inn = np.argmin(P, axis=1)
            dum = np.arange(X.shape[1])
            I_new = np.zeros((X.shape[1], k))
            I_new[dum, inn] = 1
            if np.array_equal(I, I_new):
                print("K-mean completed in epoch iteration: ", _)
                break
            I = I_new
            acc_error = P[dum, inn]
            epoch_error.append(np.mean(acc_error))

            W = I * (1 / np.sum(I, axis=0))
            restart = np.isnan(np.sum(W))
            if restart:
                m += 1
                if m > max_try:
                    return np.argmax(I, axis=1), np.array(epoch_error), W, K
                print("Again")
                break

    # print("--- %s seconds for whole process ---" % (time.time() - start_time))
    return np.argmax(I, axis=1), np.array(epoch_error), W, K


def testKmean(K1, K2, W):

    A = W.T @ K2
    B = W.T @ K1 @ W
    P = ((-2 * A)).T + np.diag(B)

    inn = np.argmin(P, axis=1)
    dum = np.arange(K2.shape[1])
    I = np.zeros((K2.shape[1], W.shape[1]))
    I[dum, inn] = 1
    return np.argmax(I, axis=1)


def findCentroid(indexes, X):

    class_number = np.max(indexes) + 1
    dict = {}
    for i in range(class_number):
        dict[str(i)] = []

    for index, x in zip(indexes, X.T):
        dict[str(index)].append(x)

    centroids = np.zeros((X.shape[0], class_number))
    for i in range(class_number):
        dum = np.array(dict[str(i)])
        dum = np.mean(dum, axis=0)
        centroids[:, i] = dum

    return centroids


def sampleX(X, y, size):

    size = int(size / 10)
    class_number = np.max(y) + 1
    dict = {}
    dict2 = {}
    for i in range(class_number):
        dict[str(i)] = []
        dict2[str(i)] = []

    i = 0
    for index, x in zip(y, X.T):
        dict[str(index)].append(x)
        dict2[str(index)].append(i)
        i += 1

    data = np.zeros((X.shape[0], int(size*10)))
    y_sample = np.zeros((int(size*10)))
    for i in range(class_number):
        dum = np.array(dict[str(i)])
        dum2 = np.array(dict2[str(i)])

        sample = np.random.choice(dum.T.shape[1], size=size, replace=False)
        data[:, i*size:(i+1)*size] = dum.T[:, sample]
        y_sample[i*size:(i+1)*size] = y[dum2[sample]]
    return data, y_sample


def findAccuracy(gt, pre):

    class_number = np.max(gt) + 1
    class_number = 10
    matrix = np.zeros((class_number, class_number))

    for g, p in zip(gt, pre):
        matrix[g, p] += 1

    total = np.sum(matrix)
    tp = 0
    for _ in range(class_number):

        ind = [int(np.floor(np.argmax(matrix) / class_number)), np.argmax(matrix) % class_number]
        tp += matrix[ind[0], ind[1]]
        matrix[ind[0], :], matrix[:, ind[1]] = 0, 0
    return np.around(tp/total, 3)


if __name__ == '__main__':

    X, y = loadlocal_mnist(images_path='train-images.idx3-ubyte', labels_path='train-labels.idx1-ubyte')
    X = X.T

    X_test, y_test = loadlocal_mnist(images_path='t10k-images.idx3-ubyte', labels_path='t10k-labels.idx1-ubyte')
    X_test = X_test.T

    # Todo: Hyperparameter Tuning
    X_sample_train, y_sample_train = sampleX(X[:, 0:30000], y[0:30000], 10000)
    X_sample_val, y_sample_val = sampleX(X[:, 30000:60000], y[30000:60000], 10000)

    #Todo: Plynomial

    print("Polynomial Hyperparameter Tuning")
    print()
    c_pol = np.linspace(-25, 25, 51)
    d = np.linspace(1, 5, 5)
    best = 0
    best_parameters = np.array([0, 0])

    for c_p in c_pol:
        for dd in d:
            # start_time = time.time()
            indexes, loss, W, K1 = trainKmean(X_sample_train, max_iter=100, k=10, kernel="Polynomial", c_poly=c_p, d=dd)
            K2 = createK2(X_sample_train, X_sample_val, type="Polynomial", c_poly=c_p, d=dd)
            test_indexes = testKmean(K1, K2, W)
            acc = findAccuracy(test_indexes.astype(int), y_sample_val.astype(int))
            print("For (c_poly , d): (" + str(c_p) + ", " + str(dd) + ") --> Accuracy: ", acc)
            print()
            # print("--- %s seconds for test ---" % (time.time() - start_time))

            if acc > best:
                best = acc
                best_parameters[0], best_parameters[1] = c_p, dd
    print("Best Parameters (c_poly , d): (" + str(best_parameters[0]) + ", " + str(best_parameters[1]) + ")")

    #Todo: Sigmoid

    print("Sigmoid Hyperparameter Tuning")
    print()
    c_sig = np.linspace(0.1, 1, 10)
    theta = np.linspace(-10, 10, 21)
    best = 0
    best_parameters = np.array([0, 0])

    for c_p in c_sig:
        for dd in theta:
            # start_time = time.time()
            indexes, loss, W, K1 = trainKmean(X_sample_train, max_iter=100, k=10, kernel="Sigmoid", c_sigmoid=c_p,
                                               theta=dd)
            K2 = createK2(X_sample_train, X_sample_val, type="Sigmoid", c_sigmoid=c_p, theta=dd)
            test_indexes = testKmean(K1, K2, W)
            acc = findAccuracy(test_indexes.astype(int), y_sample_val.astype(int))
            print("For (c_sig , theta): (" + str(c_p) + ", " + str(dd) + ") --> Accuracy: ", acc)
            print()
            # print("--- %s seconds for test ---" % (time.time() - start_time))

            if acc > best:
                best = acc
                best_parameters[0], best_parameters[1] = c_p, dd
    print("Best Parameters (c_sig , theta): (" + str(best_parameters[0]) + ", " + str(best_parameters[1]) + ")")

    # Todo: Gaussian

    print("Gaussian Hyperparameter Tuning")
    print()
    std = np.linspace(0.4, 10, 97)
    best = 0
    best_parameters = 0

    for ss in std:
        # start_time = time.time()
        indexes, loss, W, K1 = trainKmean(X_sample_train, max_iter=100, k=10, kernel="Gaussian", std=ss)
        K2 = createK2(X_sample_train, X_sample_val, type="Gaussian", std=ss)
        test_indexes = testKmean(K1, K2, W)
        acc = findAccuracy(test_indexes.astype(int), y_sample_val.astype(int))
        print("For std: " + str(ss) + " --> Accuracy: ", acc)
        print()
        # print("--- %s seconds for test ---" % (time.time() - start_time))

        if acc > best:
            best = acc
            best_parameters = ss
    print("Best Parameter std: " + str(best_parameters))

    # Todo: Show different K values
    kernels = ["No Kernel", "Sigmoid", "Polynomial", "Gaussian"]
    kk = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    std = 10
    c_sigmoid = 0.1
    theta = -5
    c_poly = -11
    d = 1

    X_sample_train, y_sample_train = sampleX(X, y, 2500)

    for kernel in kernels:
        to_plot = []
        for k in kk:
            indexes, loss, W, K1 = trainKmean(X_sample_train, max_iter=150, k=k, kernel=kernel, std=std, c_sigmoid=c_sigmoid,
                                               c_poly=c_poly, theta=theta, d=d)
            to_plot.append(loss[-1:])

        title = "Loss Function vs k for Kernel: " + str(kernel)
        plt.title(title)
        plt.plot(kk, to_plot, marker="o")
        plt.xlabel("k")
        plt.ylabel("Loss")
        plt.show()

    # Todo: Show Accuracy
    X_sample_train, y_sample_train = sampleX(X, y, 10000)
    X_sample_test, y_sample_test = sampleX(X_test, y_test, 1000)

    for kernel in kernels:
        indexes, loss, W, K1 = trainKmean(X_sample_train, max_iter=100, k=10, kernel=kernel, std=std, c_sigmoid=c_sigmoid,
                                           c_poly=c_poly, theta=theta, d=d)
        acc = findAccuracy(indexes.astype(int), y_sample_train.astype(int))
        print("Training Accuracy for Kernel: " + kernel + " --> " + str(acc))

        centroids = findCentroid(indexes, X_sample_train)
        fig, axs = plt.subplots(2, 5)
        axs = axs.flatten()
        i =0
        for centroid in centroids.T:
            axs[i].imshow(centroid.reshape(28, 28), cmap='gray')
            # plt.imshow(centroid.reshape(28, 28), cmap='gray')
            i += 1
        title = "Centroids for Kernel: " + str(kernel)
        fig.suptitle(title, fontsize=16)
        for ax in axs.flat:
            ax.label_outer()
        plt.show()

        K2 = createK2(X_sample_train, X_test, type=kernel,  std=std, c_sigmoid=c_sigmoid,
                                           c_poly=c_poly, theta=theta, d=d)
        test_indexes = testKmean(K1, K2, W)
        acc = findAccuracy(test_indexes.astype(int), y_test.astype(int))
        print("Test Accuracy for Kernel: " + kernel + " --> " + str(acc))
        print()

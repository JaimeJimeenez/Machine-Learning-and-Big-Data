from data import *

def chooseCAndSigma(X, y, x_val, y_val):
    params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(params), len(params)))
    res = []

    for v in params:

        for sigma in params:
            svm = skl.SVC(kernel='rbf', C = v, gamma = 1 / (2*sigma ** 2))
            svm.fit(X, y)
            x_pred = svm.predict(x_val)
            correct = sum(x_pred == y_val) / x_pred.shape[0] * 100
            res.append(correct)
            scores[params.index(v), params.index(sigma)] = correct
    
    return scores

def main_chooseCAndSigma():
    data = loadmat('data/ex6data3.mat')
    X = data['X']
    y = data['y'].ravel()
    x_val = data['Xval']
    y_val = data['yval'].ravel()
    scores = chooseCAndSigma(X, y, x_val, y_val)
    print(scores)
    ind = np.where(scores == scores.max())
    params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    indp1 = ind[0][0]
    indp2 = ind[1][0]

    svm = skl.SVC(kernel='rbf', C = params[indp1], gamma = 1 / (2*params[indp2] ** 2))
    svm.fit(X, y)
    border_data(x_val, y_val, svm)
    print("C = {}, sigma = {}".format(params[indp1], params[indp2]))
    
    print(ind)

def main():
    X, y = load_data(file='data/ex6data2.mat')
    print_data(X, y)

    c = 1
    sigma = 0.1
    svm = skl.SVC(kernel='rbf', C = c, gamma = (1/(2 * sigma**2)))
    svm.fit(X, y)
    border_data(X, y, svm)
    return

main_chooseCAndSigma();
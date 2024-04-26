import codecs
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.svm as skl
from sklearn.metrics import accuracy_score

from utils import *

def transformAVector(vector, dictionary):
    email = np.zeros(1899)
    for word in vector:
        if word in dictionary.keys():
            email[dictionary.get(word) - 1] += 1
    return email

def load_examples(file, num_examples):
    response = np.zeros((num_examples, 1899))
    dictionary = getVocabDict()
    for i  in range(1, num_examples + 1):
        content = codecs.open('{}/{}.txt'.format(file, str(i).zfill(4)), encoding = 'utf', errors = 'ignore').read()
        vector = email2TokenList(content)
        transform = transformAVector(vector, dictionary)
        response[i - 1] = transform
    return response

def load_data():
    eham = load_examples('data_spam/easy_ham', 2551)
    hham = load_examples('data_spam/hard_ham', 250)
    spam = load_examples('data_spam/spam', 500)
    return eham, hham, spam

def findOptimalCAndSigma(x_train, y_train, x_val, y_val):
    params = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ]
    scores = np.zeros((len(params), len(params)))
    
    for C in params:
        print('Comprobación de C en {}...'.format(C))
        for sigma in params:
            print('Comprobación de Sigma en {}...'.format(sigma))
            svm = skl.SVC(kernel='rbf', C = C, gamma = 1/(2*sigma**2))
            svm.fit(x_train, y_train)
            accuracy = accuracy_score(y_val, svm.predict(x_val))
            print('Precisión con estos parámetros: {}'.format(accuracy))
            scores[params.index(C), params.index(sigma)] = accuracy
            
    optimalC = params[np.where(scores == scores.max())[0][0]]
    optimalSigma = params[np.where(scores == scores.max())[1][0]]
    print('Se ha determinado que el C óptimo es {}'.format(optimalC))
    print('Se ha determinado que el sigma óptimo es {}'.format(optimalSigma))
    return optimalC, optimalSigma

def train(eham, hham, spam):
    yeham = np.zeros(2551)
    yhham = np.zeros(250)
    yspam = np.ones(500)
    random_state = 231052021
    
    eham_train , eham_test , yeham_train, yeham_test = train_test_split(eham,yeham, test_size = 0.25, random_state = random_state)
    hham_train , hham_test , yhham_train, yhham_test = train_test_split(hham,yhham, test_size = 0.25, random_state = random_state)
    spam_train, spam_test, yspam_train, yspam_test = train_test_split(spam,yspam, test_size = 0.25, random_state = random_state)
    
    x_train1 = np.concatenate((eham_train, hham_train, spam_train))
    y_train1 = np.concatenate((yeham_train, yhham_train, yspam_train))
    x_test = np.concatenate((eham_test, hham_test, spam_test))
    y_test = np.concatenate((yeham_test, yhham_test, yspam_test))

    xtrain, xval, ytrain, yval = train_test_split(x_train1, y_train1, test_size = 0.30, random_state = random_state)
    optimalC, optimalSigma = findOptimalCAndSigma(x_train = xtrain, y_train= ytrain, x_val = xval, y_val= yval)
    
    svmop = skl.SVC(kernel = 'rbf', C = optimalC, gamma = 1/(2*optimalSigma**2))
    svmop.fit(x_train1,y_train1)
    print(accuracy_score(y_test, svmop.predict(x_test)))

def main():
    eham, hham, spam = load_data()
    train(eham, hham, spam)

main()
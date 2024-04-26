from data import *

def main():
    X, y = load_data(file='data/ex6data1.mat')
    svm = skl.SVC(kernel='linear', C=100) # C = 100
    svm.fit(X, y)
    border_data(X, y, svm)
    
main()
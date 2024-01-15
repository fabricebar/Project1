import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
from IPython.display import display

# To keep consistency and to be able to compare runs the randomness is kept constant.
np.random.seed(7)

# Make data with Franke function which includes the error e.
def FrankeFunction(x,y,e):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    if len(x.shape)>1:
        return term1 + term2 + term3 + term4 + (e*np.random.randn(len(x),len(x)))
    else:
        return term1 + term2 + term3 + term4 + (e*np.random.randn(len(x)))

# Makes the designmatrix using x and y as input also polydegree n.
def dX(x,y,n):
    
    if len(x.shape)>1:
        x=x.ravel()
        y=y.ravel()

    l=int((n+1)*(n+2)/2)
    X=np.ones((len(x),l))
    for i in range(1,n+1):
        q=int(i*(i+1)/2)
        for j in range(i+1):
            X[:,q+j]=(x**(i-j))*(y**j)
    return X

# Makes the coefficents for OLS estimator.
def beta_OLS(X,z):
    beta= np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)
    return beta

# Makes the coefficents for Ridge estimator.
def beta_R(X,z,lamda):
    beta= np.linalg.pinv(X.T.dot(X)+lamda*np.identity((X.shape[1]))).dot(X.T).dot(z)
    return beta
    pass

# Setting up input parameters.
N=200 # Number of points
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
x,y=np.meshgrid(x,y)
z = FrankeFunction(x, y, 0.1)
Y=z.ravel()

# Calculates the MSE.
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# Calculates the R2.
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

# Plots the R2 and MSE as a function of polynimial degree.
def plot_mse(n,t):
    p=np.arange(1,n+1,1)
    mse_train=np.zeros(len(p))
    r2_train=np.zeros(len(p))
    mse_test=np.zeros(len(p))
    r2_test=np.zeros(len(p))

    mse2_train=np.zeros(len(p))
    r22_train=np.zeros(len(p))
    mse2_test=np.zeros(len(p))
    r22_test=np.zeros(len(p))

    z_shape=z.shape
    for i in range(len(p)):
        X2=dX(x,y,i+1)

        X2_train, X2_test, Y_train, Y_test= train_test_split(X2,Y,test_size=0.2)

        X2_scaler = StandardScaler(with_std=True)
        X2_scaler.fit(X2_train)
        X2_train = X2_scaler.transform(X2_train)
        X2_test = X2_scaler.transform(X2_test)

        Y_scaler = StandardScaler()
        Y_scaler.fit(Y_train.reshape(-1,1))
        Y_train = (Y_scaler.transform(Y_train.reshape(-1,1))).ravel()
        Y_test = (Y_scaler.transform(Y_test.reshape(-1,1))).ravel()

        if t==0:
            model1="OLS"
            model2="OLS X2"
            
            b2=beta_OLS(X2_train,Y_train)

            z2_tilde=(X2_train@b2)
            z2_pred=(X2_test@b2)

            mse2_train[i]=MSE(Y_train,z2_tilde)
            r22_train[i]=R2(Y_train,z2_tilde)
            mse2_test[i]=MSE(Y_test,z2_pred)
            r22_test[i]=R2(Y_test,z2_pred)
            
        else:
            model1="Ridge"

            b=beta_OLS(X2_train,Y_train)
            b2=beta_R(X2_train,Y_train,t)

            model2="Lasso"
            RegLasso = linear_model.Lasso(t,fit_intercept=False)
            RegLasso.fit(X2_train,Y_train)

            z_tilde=X2_train@b2
            z_pred=X2_test@b2

            z2_tilde=(RegLasso.predict(X2_train))
            z2_pred=(RegLasso.predict(X2_test))

            mse_train[i]=MSE(Y_train,z_tilde)
            r2_train[i]=R2(Y_train,z_tilde)
            mse_test[i]=MSE(Y_test,z_pred)
            r2_test[i]=R2(Y_test,z_pred)

            mse2_train[i]=MSE(Y_train,z2_tilde)
            r22_train[i]=R2(Y_train,z2_tilde)
            mse2_test[i]=MSE(Y_test,z2_pred)
            r22_test[i]=R2(Y_test,z2_pred)

        data={"β_OLS estimator":b,"β_Ridge estimator":b2,"β_Lasso estimator":RegLasso.coef_}
        XPandas = pd.DataFrame(data)

        data2={"β_Ridge estimator":b2}
        XPandas2 = pd.DataFrame(data2)

        data3={"β_OLS estimator":b}
        XPandas3 = pd.DataFrame(data3)

        print()
        print()
        display(XPandas)
        print()

    plt.plot(p,(mse2_train),label="MSE_train")
    plt.plot(p,(mse2_test),label="MSE_test")
    plt.plot(p,(r22_train),label="R2_train")
    plt.plot(p,(r22_test),label="R2_test")
    plt.title(f"{model2} MSE2 & R2-Score & lamda = {t}")
    plt.xlabel("polynomial degree")
    plt.ylabel("score")
    plt.legend()
    plt.show()

    plt.plot(p,(mse_train),label="MSE_train")
    plt.plot(p,(mse_test),label="MSE_test")
    plt.plot(p,(r2_train),label="R2_train")
    plt.plot(p,(r2_test),label="R2_test")
    plt.title(f"{model1} MSE2 & R2-Score &  λ = {t}")
    plt.xlabel("polynomial degree")
    plt.ylabel("score")
    plt.legend()
    plt.show()

# Plots the R2 and MSE as a function of the hyperperameter lambda.
def plot_lamda(n,t):
    lamdas_ridge=np.logspace(-2,7,t)
    lamdas_lasso=np.logspace(-4,0,t)

    Rmse_train=np.zeros(len(lamdas_ridge))
    Rr2_train=np.zeros(len(lamdas_ridge))
    Rmse_test=np.zeros(len(lamdas_ridge))
    Rr2_test=np.zeros(len(lamdas_ridge))

    Lmse_train=np.zeros(len(lamdas_lasso))
    Lr2_train=np.zeros(len(lamdas_lasso))
    Lmse_test=np.zeros(len(lamdas_lasso))
    Lr2_test=np.zeros(len(lamdas_lasso))

    for i in range(len(lamdas_ridge)):
        X=dX(x,y,n)

        X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2,random_state=4)

        X_scaler = StandardScaler()
        X_scaler.fit(X_train)
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)

        Y_scaler = StandardScaler()
        Y_scaler.fit(Y_train.reshape(-1,1))
        Y_train = (Y_scaler.transform(Y_train.reshape(-1,1))).ravel()
        Y_test = (Y_scaler.transform(Y_test.reshape(-1,1))).ravel()

        beta=beta_R(X_train,Y_train,lamdas_ridge[i])

        RegLasso = linear_model.Lasso(lamdas_lasso[i],fit_intercept=False)
        RegLasso.fit(X_train,Y_train)

        z_tilde=X_train@beta
        z_pred=X_test@beta

        z2_tilde=(RegLasso.predict(X_train))
        z2_pred=(RegLasso.predict(X_test))

        Rmse_train[i]=MSE(Y_train,z_tilde)
        Rr2_train[i]=R2(Y_train,z_tilde)
        Rmse_test[i]=MSE(Y_test,z_pred)
        Rr2_test[i]=R2(Y_test,z_pred)

        Lmse_train[i]=MSE(Y_train,z2_tilde)
        Lr2_train[i]=R2(Y_train,z2_tilde)
        Lmse_test[i]=MSE(Y_test,z2_pred)
        Lr2_test[i]=R2(Y_test,z2_pred)

    plt.plot(np.log10(lamdas_ridge),(Rmse_train),label="MSE_train")
    plt.plot(np.log10(lamdas_ridge),(Rmse_test),label="MSE_test")
    plt.plot(np.log10(lamdas_ridge),(Rr2_train),label="R2_train")
    plt.plot(np.log10(lamdas_ridge),(Rr2_test),label="R2_test")
    plt.title(f"Ridge MSE & R2-Score polydegree = {n} ")
    plt.xlabel("log10 lamdas")
    plt.ylabel("score")
    plt.legend()
    plt.show()

    plt.plot(np.log10(lamdas_lasso),(Lmse_train),label="MSE_train")
    plt.plot(np.log10(lamdas_lasso),(Lmse_test),label="MSE_test")
    plt.plot(np.log10(lamdas_lasso),(Lr2_train),label="R2_train")
    plt.plot(np.log10(lamdas_lasso),(Lr2_test),label="R2_test")
    plt.title(f"Lasso MSE2 & R2-Score polydegree = {n} ")
    plt.xlabel("log10 lamdas")
    plt.ylabel("score")
    plt.legend()
    plt.show()

# Plots the error, bias and variance vs complexity using bootstrap resampling
def bootstrap(n,n_b):

    p=np.arange(1,n+1,1)
    error=np.zeros(len(p))
    error2=np.zeros(len(p))
    bias=np.zeros(len(p))
    variance=np.zeros(len(p))

    error12=np.zeros(len(p))
    error22=np.zeros(len(p))
    bias2=np.zeros(len(p))
    variance2=np.zeros(len(p))

    bootstraps=n_b
    mse_t=np.zeros(len(p))
    for i in range(len(p)):
        X=dX(x,y,i+1)

        X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2)

        X_scaler = StandardScaler()
        X_scaler.fit(X_train)
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)

        Y_scaler = StandardScaler()
        Y_scaler.fit(Y_train.reshape(-1,1))
        Y_train = (Y_scaler.transform(Y_train.reshape(-1,1))).ravel()
        Y_test = (Y_scaler.transform(Y_test.reshape(-1,1))).ravel()

        z_tilde=np.empty((Y_train.shape[0],bootstraps))
        z_pred=np.empty((Y_test.shape[0],bootstraps))
        for j in range(bootstraps):
            X_, z_ =resample(X_train,Y_train)
            beta=beta_OLS(X_,z_)
            z_tilde[:,j]=X_@beta
            z_pred[:,j]=(X_test@beta)

        error2[i]=np.mean( np.mean((Y_train[:,np.newaxis] - z_tilde)**2, axis=1, keepdims=True) )
        error[i]=np.mean( np.mean((Y_test[:,np.newaxis] - z_pred)**2, axis=1, keepdims=True) )
        bias[i]=np.mean( (Y_test[:,np.newaxis] - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance[i]=np.mean( np.var(z_pred, axis=1, keepdims=True) )

    plt.plot(p,error,label="test error")
    plt.plot(p,bias, label="bias")
    plt.plot(p,variance, label="variance")
    plt.title(f"Bias-variance tradeoff bootstraps={n_b}")
    plt.xlabel("polynomial degree")
    plt.legend()
    plt.show()

# Plots the error vs complexity using k-fold CV
def CrossV(n,k,nlamdas,t):
    p=np.arange(1,n+1,1)
    error=np.zeros(len(p))
    estimated_mse_KFold=np.zeros(len(p))
    estimated_mse_KFold_train=np.zeros(len(p))
    estimated_mse_sk_KFold=np.zeros(len(p))

    if t==1:
        for i in range(len(p)):
            X=dX(x,y,i+1)

            kfold = KFold(n_splits = k)
            "scores_KFold = np.zeros((5, k))"
            scores_KFold = []
            scores_KFold_train = []

            for train_inds, test_inds in kfold.split(X):
                Xtrain = X[train_inds]
                Ytrain = Y[train_inds]

                Xtest = X[test_inds]
                Ytest = Y[test_inds]

                X_scaler = StandardScaler(with_std=False)
                X_scaler.fit(Xtrain)
                Xtrain = X_scaler.transform(Xtrain)
                Xtest = X_scaler.transform(Xtest)

                Y_scaler = StandardScaler(with_std=False)
                Y_scaler.fit(Ytrain.reshape(-1,1))
                Ytrain = (Y_scaler.transform(Ytrain.reshape(-1,1))).ravel()
                Ytest = (Y_scaler.transform(Ytest.reshape(-1,1))).ravel()


                beta=beta_OLS(Xtrain,Ytrain)
                z_pred=Xtest@beta
                z_tilde=Xtrain@beta
                scores_KFold.append(MSE(Ytest,z_pred))
                scores_KFold_train.append(MSE(Ytrain,z_tilde))

            scores_KFold=np.array(scores_KFold)
            scores_KFold_train=np.array(scores_KFold_train)

            OLS = LinearRegression(fit_intercept=False)
            scores_sk_Kfold = cross_val_score(OLS, X, Y, scoring='neg_mean_squared_error', cv=kfold)
            estimated_mse_KFold[i] = np.mean(scores_KFold, axis = 0)
            estimated_mse_KFold_train[i] = np.mean(scores_KFold_train, axis = 0)
            estimated_mse_sk_KFold[i] = np.mean(-scores_sk_Kfold , axis = 0)

        plt.plot(p,(estimated_mse_KFold_train),label="training error")
        plt.plot(p,(estimated_mse_KFold),label="test error")
        plt.plot(p,(estimated_mse_sk_KFold),label="test error sk",linestyle=":",color="black")
        plt.xlabel("polydegree")
        plt.ylabel("MSE")
        plt.title(f"Estimated MSE {k}-fold CV OLS")
        plt.legend()
        plt.show()

    elif t==2:
        lamdas=np.logspace(-6,6,nlamdas)
        estimated_mse_KFold_test=np.zeros(len(lamdas))
        estimated_mse_KFold_train=np.zeros(len(lamdas))
        estimated_mse_sk_KFold=np.zeros(len(lamdas))

        for i in range(nlamdas):
            X=dX(x,y,n)
            kfold = KFold(n_splits = k)
            scores_KFold_test = []
            scores_KFold_train = []
            for train_inds, test_inds in kfold.split(X):
                Xtrain = X[train_inds]
                Ytrain = Y[train_inds]
                Xtest = X[test_inds]
                Ytest = Y[test_inds]

                X_scaler = StandardScaler(with_std=False)
                X_scaler.fit(Xtrain)
                Xtrain = X_scaler.transform(Xtrain)
                Xtest = X_scaler.transform(Xtest)

                Y_scaler = StandardScaler(with_std=False)
                Y_scaler.fit(Ytrain.reshape(-1,1))
                Ytrain = (Y_scaler.transform(Ytrain.reshape(-1,1))).ravel()
                Ytest = (Y_scaler.transform(Ytest.reshape(-1,1))).ravel()

                beta=beta_R(Xtrain, Ytrain,lamdas[i])
                z_pred=Xtest@beta
                z_tilde=Xtrain@beta

                scores_KFold_test.append(MSE(Ytest,z_pred))
                scores_KFold_train.append(MSE(Ytrain,z_tilde))

            scores_KFold_test=np.array(scores_KFold_test)
            scores_KFold_train=np.array(scores_KFold_train)

            ridge = Ridge(alpha=lamdas[i],fit_intercept=True)
            scores_sk_Kfold = cross_val_score(ridge, X, Y, scoring='neg_mean_squared_error', cv=kfold)
            estimated_mse_KFold_test[i] = np.mean(scores_KFold_test, axis = 0)
            estimated_mse_KFold_train[i] = np.mean(scores_KFold_train, axis = 0)
            estimated_mse_sk_KFold[i] = np.mean(-scores_sk_Kfold , axis = 0)

        
        plt.plot(np.log10(lamdas),(estimated_mse_KFold_train),label="train error")
        plt.plot(np.log10(lamdas),(estimated_mse_KFold_test),label="test error")
        plt.plot(np.log10(lamdas),(estimated_mse_sk_KFold),label="test error sk",linestyle=":")
        plt.xlabel("log10 lamdas")
        plt.ylabel("MSE")
        plt.title(f"Estimated MSE of {k}-fold CV of polydegree {n} Ridge")
        plt.legend()
        plt.show()

    elif t==3:
        lamdas=np.logspace(-4,2,nlamdas)
        estimated_mse_KFold_test=np.zeros(len(lamdas))
        estimated_mse_KFold_train=np.zeros(len(lamdas))
        estimated_mse_sk_KFold=np.zeros(len(lamdas))

        for i in range(nlamdas):
            X=dX(x,y,n)
            kfold = KFold(n_splits = k)
            scores_KFold_test = []
            scores_KFold_train = []

            for train_inds, test_inds in kfold.split(X):
                Xtrain = X[train_inds]
                Ytrain = Y[train_inds]
                Xtest = X[test_inds]
                Ytest = Y[test_inds]

                X_scaler = StandardScaler(with_std=False)
                X_scaler.fit(Xtrain)
                Xtrain = X_scaler.transform(Xtrain)
                Xtest = X_scaler.transform(Xtest)

                Y_scaler = StandardScaler(with_std=False)
                Y_scaler.fit(Ytrain.reshape(-1,1))
                Ytrain = (Y_scaler.transform(Ytrain.reshape(-1,1))).ravel()
                Ytest = (Y_scaler.transform(Ytest.reshape(-1,1))).ravel()

                RegLasso = linear_model.Lasso(lamdas[i],fit_intercept=True)
                RegLasso.fit(Xtrain,Ytrain)

                z_pred=RegLasso.predict(Xtest)
                z_tilde=RegLasso.predict(Xtrain)

                scores_KFold_test.append(MSE(Ytest,z_pred))
                scores_KFold_train.append(MSE(Ytrain,z_tilde))

            scores_KFold_test=np.array(scores_KFold_test)
            scores_KFold_train=np.array(scores_KFold_train)

            scores_sk_Kfold = cross_val_score(RegLasso, X, Y, scoring='neg_mean_squared_error', cv=kfold)

            estimated_mse_KFold_test[i] = np.mean(scores_KFold_test, axis = 0)
            estimated_mse_KFold_train[i] = np.mean(scores_KFold_train, axis = 0)
            estimated_mse_sk_KFold[i] = np.mean(-scores_sk_Kfold , axis = 0)


        plt.plot(np.log10(lamdas),(estimated_mse_KFold_train),label="train error")
        plt.plot(np.log10(lamdas),(estimated_mse_KFold_test),label="test error")
        plt.plot(np.log10(lamdas),(estimated_mse_sk_KFold),label="test error sk",linestyle=":")
        plt.xlabel("log10 lamdas")
        plt.ylabel("MSE")
        plt.title(f"Estimated MSE of {k}-fold CV of polydegree {n} LASSO")
        plt.legend()
        plt.show()

# Plots the OLS error vs datapoints/noise using bootstrap
def data_noise(N,e,n,t,n_b):
    if t==0:
        e=np.arange(0,e,0.05)
        mse2_train=np.zeros(len(e))
        mse2_test=np.zeros(len(e))

        error=np.zeros(len(e))
        error2=np.zeros(len(e))
        bias=np.zeros(len(e))
        variance=np.zeros(len(e))

        bootstraps=n_b
        for i in range(len(e)):
            x = np.linspace(0, 1, N)
            y = np.linspace(0, 1, N)
            x,y=np.meshgrid(x,y)
            z = FrankeFunction(x, y, e[i])
            print(f"dette er e = {(e[i])}")
            Y=z.ravel()

            X2=dX(x,y,n)

            X2_train, X2_test, Y_train, Y_test= train_test_split(X2,Y,test_size=0.2)

            X2_scaler = StandardScaler(with_std=True)
            X2_scaler.fit(X2_train)
            X2_train = X2_scaler.transform(X2_train)
            X2_test = X2_scaler.transform(X2_test)

            Y_scaler = StandardScaler()
            Y_scaler.fit(Y_train.reshape(-1,1))
            Y_train = (Y_scaler.transform(Y_train.reshape(-1,1))).ravel()
            Y_test = (Y_scaler.transform(Y_test.reshape(-1,1))).ravel()


            b2=beta_OLS(X2_train,Y_train)

            z2_tilde=(X2_train@b2)
            z2_pred=(X2_test@b2)

            mse2_train[i]=MSE(Y_train,z2_tilde)
            mse2_test[i]=MSE(Y_test,z2_pred)

        plt.plot(e,(mse2_train),label="MSE_train")
        plt.plot(e,(mse2_test),label="MSE_test")

        plt.title(f" MSE vs Noise of Polydegree = {n} and Datapoints = {N}")
        plt.xlabel("Noise")
        plt.ylabel("error")
        plt.legend()
        plt.show()
    else:
        N=np.arange(20,N,5)
        mse2_train=np.zeros(len(N))
        mse2_test=np.zeros(len(N))

        error=np.zeros(len(N))
        error2=np.zeros(len(N))
        bias=np.zeros(len(N))
        variance=np.zeros(len(N))

        bootstraps=n_b
        mse_t=np.zeros(len(N))

        for i in range(len(N)):
            x = np.linspace(0, 1, N[i])
            y = np.linspace(0, 1, N[i])
            print(f"dette er x = {len(x)}")
            x,y=np.meshgrid(x,y)
            z = FrankeFunction(x, y, e)
            Y=z.ravel()

            X2=dX(x,y,n)

            X2_train, X2_test, Y_train, Y_test= train_test_split(X2,Y,test_size=0.2)

            X2_scaler = StandardScaler(with_std=True)
            X2_scaler.fit(X2_train)
            X2_train = X2_scaler.transform(X2_train)
            X2_test = X2_scaler.transform(X2_test)

            Y_scaler = StandardScaler()
            Y_scaler.fit(Y_train.reshape(-1,1))
            Y_train = (Y_scaler.transform(Y_train.reshape(-1,1))).ravel()
            Y_test = (Y_scaler.transform(Y_test.reshape(-1,1))).ravel()

            z_tilde=np.empty((Y_train.shape[0],bootstraps))
            z_pred=np.empty((Y_test.shape[0],bootstraps))
            for j in range(bootstraps):
                X_, z_ =resample(X2_train,Y_train)
                beta=beta_OLS(X_,z_)
                z_tilde[:,j]=X_@beta
                z_pred[:,j]=(X2_test@beta)

            error2[i]=np.mean( np.mean((Y_train[:,np.newaxis] - z_tilde)**2, axis=1, keepdims=True) )
            error[i]=np.mean( np.mean((Y_test[:,np.newaxis] - z_pred)**2, axis=1, keepdims=True) )
            bias[i]=np.mean( (Y_test[:,np.newaxis] - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance[i]=np.mean( np.var(z_pred, axis=1, keepdims=True) )

            b2=beta_OLS(X2_train,Y_train)

            z2_tilde=(X2_train@b2)
            z2_pred=(X2_test@b2)

            mse2_train[i]=MSE(Y_train,z2_tilde)
            mse2_test[i]=MSE(Y_test,z2_pred)
            print(f"dette er i = {i}")
            print()

        plt.plot(N,(mse2_train),label="MSE_train")
        plt.plot(N,(mse2_test),label="MSE_test")

        plt.title(f" MSE vs Datapoints of polydegree = {n} and noise = {e}")
        plt.xlabel("Number of datapoints")
        plt.ylabel("error")
        plt.legend()
        plt.show()


"plot_mse(13,10)"

"plot_lamda(3,60)"

"bootstrap(10,110)"

"CrossV(5,10,60,2)"

data_noise(150,0.1,25,1,50)

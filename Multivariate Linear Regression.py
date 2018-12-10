import numpy as np
import matplotlib.pyplot as plt

def join_ones(X):
	if X.shape[0]==X.size:
		X=[[i] for i in X.tolist()]
	else:
		X=X.tolist()
	X_=[]
	for i in X:
		a=[1,]
		for n in list(i):
			a.append(n)

		X_.append(a)
	return np.array(X_)

class LinearRegression:

        def __init__(self,epochs=100,alpha=0.001,lambda_=0.01):
                self.epochs=epochs
                self.alpha=alpha
                self.lambda_=lambda_

        def __train__(self,X=None,y=None,givenXy=False,learning_graph=False,lambda_=False):
                if not givenXy:
                        X=self.X
                        y=self.y
                if lambda_:
                        self.lambda_=lambda_
                
                m,n=X.shape
                self.theta=np.ones(n)
                self.h0=lambda x: np.dot(x,self.theta)
                error=[]

                for i in range(self.epochs):
                        self.J=(np.sum((self.h0(X)-y)**2)+np.sum(self.theta[1:]**2)*self.lambda_)/(2*m)
                        gradient0=np.sum(self.h0(X)-y)/m
                        gradientj=np.sum((self.h0(X)-y)*X[:,1:].transpose())/m + (self.lambda_/m)*np.sum(self.theta[1:])
                        self.theta[0]=self.theta[0]-self.alpha*gradient0
                        self.theta[1:]=self.theta[1:]-self.alpha*gradientj
                        error.append(self.J)

                if learning_graph:
                        plt.plot(range(self.epochs),error)
                        plt.xlabel('Iterations')
                        plt.ylabel('Squared Error')
                        plt.title('Iterations vs Error')
                        plt.legend(['Error'])
                        plt.show()

                return self.theta,self.J

        def predict(self,X):
                if self.polynomial_features and self.polynomial_features_degree>=2:
                        X=self.__PolynomialFeatures__(X,self.polynomial_features_degree)
                X=join_ones(X)
                return self.h0(X)

        def __getJ__(self,X,y,lambda_=False,add_ones=True):
                if not lambda_:
                        lambda_=self.lambda_
                if add_ones:
                        X=join_ones(X)
                m,n=X.shape
                J=(np.sum((self.h0(X)-y)**2)+np.sum(self.theta[1:]**2)*lambda_)/(2*m)
                return J

        def __chooseLambda__(self,cvX,cvY,step=0.1,multiplier=1,initial=0,upto=2,graph=True):
                lambda_=initial
                all_lambda=[]
                trainJ=[]
                cvJ=[]
                
                while lambda_<=upto:
                        self.lambda_=lambda_
                        theta,J=self.__train__()
                        trainJ.append(J)
                        cvJ.append(self.__getJ__(cvX,cvY,lambda_=0,add_ones=False))
                        all_lambda.append(lambda_)
                        lambda_*=multiplier
                        lambda_+=step
                        

                self.lambda_=round(all_lambda[cvJ.index(min(cvJ))],3)
                cvE=round(min(cvJ),2)
                tE=round(trainJ[cvJ.index(min(cvJ))],2)
                
                print('Lambda: {0}\t 	 Training Error: {1}\t CV Error: {2}'.format(self.lambda_,tE,cvE))

                if graph:
                        plt.plot(all_lambda,trainJ)
                        plt.plot(all_lambda,cvJ)
                        plt.legend(['Training Error','Cross Validation Error'])
                        plt.xlabel('Lambda')
                        plt.ylabel('Error')
                        plt.title('Choosing Lambda')
                        plt.show()

        def __chooseAlpha__(self,cvX,cvY,step=0.001,multiplier=1,initial=0,upto=1,graph=True):
                alpha=initial
                all_alpha=[]
                trainJ=[]
                cvJ=[]
                
                while alpha<=upto:
                        self.alpha=round(alpha,4)
                        theta,J=self.__train__()
                        trainJ.append(J)
                        cvJ.append(self.__getJ__(cvX,cvY,lambda_=0,add_ones=False))
                        all_alpha.append(alpha)
                        alpha*=multiplier
                        alpha+=step

                self.alpha=all_alpha[cvJ.index(min(cvJ))]
                cvE=round(min(cvJ),2)
                tE=round(trainJ[cvJ.index(min(cvJ))],2)
                
                print('Learning Rate: {0}\t Training Error: {1}\t CV Error: {2}'.format(self.alpha,tE,cvE))

                if graph:
                        plt.plot(all_alpha,trainJ)
                        plt.plot(all_alpha,cvJ)
                        plt.legend(['Training Error','Cross Validation Error'])
                        plt.xlabel('Learning Rate')
                        plt.ylabel('Error')
                        plt.title('Choosing Learning Rate')
                        plt.show()
        def __learningCurve__(self,testX,testY,learning_curve_multiplier=1,learning_curve_step=1):
                X=self.X
                y=self.y
                n=1
                TrainingError=[]
                N=[]
                TestError=[]
                
                while n<=X.shape[0]:
                        theta,j=self.__train__(X=X[:n,:],y=y[:n],givenXy=True,lambda_=0)
                        TrainingError.append(j)
                        N.append(n)
                        jT=self.__getJ__(testX,testY,lambda_=0,add_ones=False)
                        TestError.append(jT)
                        n*=learning_curve_multiplier
                        n+=learning_curve_step
                        
                plt.plot(N,TrainingError)
                plt.plot(N,TestError)
                plt.ylabel('Error')
                plt.xlabel('Number of training examples (m)')
                plt.legend(['Training Error','Test Error'])
                plt.show()
        def __PolynomialFeatures__(self,X,degree):
                try:
                        m,n=X.shape
                except:
                        n=1
                X_=[]
                X_.append(X)
                for power in range(2,degree+1):
                        if n==1:
                            X_.append(X**power)
                            
                        else:
                            K=[]
                            for i in X: 
                                K.append(i**power)
                            X_.append(K)
                if n!=1:
                        X_=np.concatenate(tuple(X_),axis=1).transpose()
                else:
                        X_=np.array(X_)
                return np.array(X_).transpose()
                        
        def fit(self,X,y,test_size=0.1,cv_size=0.1,chooseLambda=False,
                Test=True,
                chooseAlpha=False,lambda_step=0.1,lambda_multiplier=1,lambda_initial=0,
                lambda_upto=2,lambda_graph=False,
                alpha_step=0.001,alpha_multiplier=1,alpha_initial=0.0,alpha_upto=0.01,
                alpha_graph=False,
                learning_graph=False,
                shuffle=False,shuffle_random_state=100,
                learning_curve=False,learning_curve_multiplier=1,learning_curve_step=1,
                polynomial_features=False,polynomial_features_degree=2):

                self.polynomial_features=polynomial_features
                self.polynomial_features_degree=polynomial_features_degree
                
                if shuffle:
                        from sklearn.utils import shuffle
                        X,y=shuffle(X,y,random_state=shuffle_random_state)
                if polynomial_features and polynomial_features_degree>=2:
                        X=self.__PolynomialFeatures__(X,degree=polynomial_features_degree)
        
                self.X=join_ones(X)
                self.y=y
                if Test:
                        trainX,testX,trainY,testY=train_test_split(self.X,self.y,test_size=test_size+cv_size,random_state=100)
                        testX,cvX,testY,cvY=train_test_split(testX,testY,test_size=cv_size/(cv_size+test_size))
                        self.X=trainX;self.y=trainY
                if chooseAlpha:
                    self.__chooseAlpha__(cvX,cvY,step=alpha_step,multiplier=alpha_multiplier,
                                         initial=alpha_initial,upto=alpha_upto,graph=alpha_graph)
                if learning_curve:
                    self.__learningCurve__(testX,testY,learning_curve_multiplier,learning_curve_step)
                
                if chooseLambda:
                    self.__chooseLambda__(cvX,cvY,step=lambda_step,multiplier=lambda_multiplier,
                                         initial=lambda_initial,upto=lambda_upto,graph=lambda_graph)
                
                self.theta,trainError=self.__train__(learning_graph=learning_graph)

                trainError=round(trainError,2)
                if Test:
                        cvError=round(self.__getJ__(cvX,cvY,add_ones=False),2)
                        testError=round(self.__getJ__(testX,testY,add_ones=False),2)
                        print('\n')
                        print('      CV Error:',cvError)
                        print('Training Error:',trainError)
                        print('    Test Error:',testError)
                else:
                        print('\nTraining Error:',trainError)
               

                return self.theta

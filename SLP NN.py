import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SLP:
    def __init__(self,w,lr,epochs):
        self.w=w
        self.lr=lr
        self.epochs=epochs
        
    #@activation_function
    def binary_active(self,prob):
        if self.prob>0:
            return 1
        else:
            return 0
    def probagation_func(self,x):
        self.prob=self.w.dot(x.T)
        return self.prob
    
    def feed_forward(self,x,y):
        self.y_pred=self.binary_active(self.probagation_func(x))
        self.delta_term=y-self.y_pred
        if self.delta_term !=0:
            self.w+=(self.lr*self.delta_term*x)
            
    def training(self,x,y):
        for p in range(self.epochs):
            print('================= epoch(',p,')===============')
            print('W:',self.w)
            for i in range(len(x)):
                self.rand_num=np.random.randint(len(x))
                self.instance_x=x[self.rand_num]
                self.instance_y=y[self.rand_num]
                self.feed_forward(self.instance_x,self.instance_y)
            

    def testing(self,x_test):
        self.y_pred=[]
        for i in range(len(x_test)):
            self.y_pred.append(self.binary_active(self.probagation_func(x_test[i])))
        return self.y_pred
    
    
#dataset=pd.DataFrame([[0,0,0],[1,1,1],[1,0,1],[0,1,1]])
        
dataset=pd.read_csv('Dataset1.csv',header=None)
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import *
s=StandardScaler()
x=s.fit_transform(x)

from sklearn.model_selection import *

x=np.append(np.ones((len(x),1)),x,axis=1)
w=np.array(np.random.rand(1,len(x[0])))
#w=np.array(np.zeros((1,len(x[0]))))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)


s=SLP(w,0.1,100)
s.training(x_train,y_train)

pred=s.testing(x_test)

accuracy=(np.sum(pred==y_test)/len(y_test))*100

x0=x_test[y_test==0]
x1=x_test[y_test==1]

plt.scatter(x0[:,1],x0[:,2],color='red')
plt.scatter(x1[:,1],x1[:,2],color='green')

x_a=np.arange(-3,2.5,0.5)
y_a=-(w[0,0]+w[0,1]*x_a)/w[0,2]

plt.plot(x_a,y_a,color='blue')




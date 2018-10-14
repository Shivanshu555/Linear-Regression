#Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *

#Loading the data 
root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("text files","*.txt")))
data=pd.read_csv(root.filename)
root.destroy()

#Showing the attributes for easy access
print('Attributes find are: ')
print(','.join(data.columns))

#Taking the value of X (dependent)
X_name=input('\nEnter the name of X: ')
initial_X=list(data.get(X_name))

#Taking the value of y (independent)
y_name=input('Enter the name of y: ')
initial_y=list(data.get(y_name))

#Building a matrix of X and y
X=np.matrix([[1,i] for i in initial_X])
y=np.matrix([[i] for i in initial_y])

#Calculating the value of theta using normal equation
def thetas(X,y):
    theta = np.linalg.pinv(X.transpose()*X) * X.transpose() * y
    return theta
theta=thetas(X,y)

#Displaying the relation
m=float(theta[1][0])
b=float(theta[0][0])
print('Relation found: y = {0}x+ {1}'.format(m,b))

#Showing the plot for accuracy visualization
plt.scatter(initial_X,initial_y)
plt.plot(initial_X,[m*x+b for x in initial_X])
plt.xlabel(X_name)
plt.ylabel(y_name)
plt.title("{0} relation with respect to {1}".format(y_name,X_name))
plt.show()

#Asking to save the plot          
yn=input('Do you want to save the figure [y|n]')
if yn.lower()=='y':

    #Getting the directory to save plot
    root=Tk()
    directory=filedialog.askdirectory()
    root.destroy()
    name=input('Enter the name for your plot: ')
    name= directory+'\\' + name+ '.jpg'

    #Recreating the plot
    plt.scatter(initial_X,initial_y)
    plt.plot(initial_X,[m*x+b for x in initial_X])
    plt.xlabel(X_name)
    plt.ylabel(y_name)
    plt.title("{0} relation with respect to {1}".format(y_name,X_name))
    plt.savefig(name)
          
else:
    #Exiting the program
    exit()
    

##This program was created by Ian Montgomery for ISTA 421

import numpy as np
import matplotlib.pyplot as plt

##a_1 = np.asmatrix(np.loadtxt("./code/data/womens100.csv",delimiter=','))  ##Womens100 data
##a_1 = np.asmatrix(np.loadtxt("./code/data/olympics.csv",delimiter=','))  ##Mens Olympics data
a_1 = np.asmatrix(np.loadtxt("./code/data/synthdata.csv",delimiter = ",")) ##Synthetic data
print(a_1)


(N,M) = a_1.shape  ##finds dimensions of the data

##print(N) ## debug

order_Poly = raw_input("What order regression would you like :")

order_Poly = int(order_Poly)
A_x = []
B = a_1[:,1]
error_plot = []
##print(A_x)


for j in range(0,order_Poly+1):  ## this is used to create the matricied for the regression model
    A = np.array(a_1[0,0]**(j))
    for i in range(1,N):
        A = np.append(A,a_1[i,0]**(j))
    
##    print(A)

    A_x = np.append(A_x,A,axis=1)
    A_m = np.array(A_x)
    A_m.resize(j+1,N)
    At = A_m  ##easier then doing the transpose agian later
    A_m = np.transpose(A_m)
##    print(A_m)
##    print('\n\n\n\n')

####This area would be represented in Matlab as inv(A'*A)*A'*B which is the regression model we use

    AtA = np.dot(At,A_m)
    invAtA = np.linalg.inv(AtA)
    AtB = np.dot(At,B)
    fit = np.dot(invAtA,AtB)
    print(fit)
    print('\n\n\n\n\n')
    pos_x = []
    pos_y = []
    pos_ye = []

    for x in range(0,N):
        pos_x.append(a_1[x,0])

        
####  This area was an attemped at Mean Square Error
##    pos_x = np.sort(pos_x)
##    for x in range(0,N):
##        y = 0
##        y_e = 0
##        for k in range(0,j+1):
##            t = fit[k,0]*(pos_x[x] ** k)
##            t_e = fit[k,0]*(a_1[x,1] ** k)
##            y = y + t
##            y_e = y_e + t_e
##
##        pos_y.append(y)
##        pos_ye.append(y_e)
##        
##    pos = np.array((pos_x,pos_y))
##    pos = np.transpose(pos)

##    sum_sqr_delta = 0
##    for e in range(0,N):
##        delta_est = pos_ye[e]- a_1[e,1]
##       print(delta_est)
##        sqr_delta = delta_est
##        sum_sqr_delta = sum_sqr_delta + sqr_delta
##
##    sum_sqr_e = sum_sqr_delta/N
##
##    print(sum_sqr_e)
##    error_plot.append(sum_sqr_e)
##
####  from what I can see, at a certain point the limitation of the floating point, models at a cerain point no longer match, but it does depend on the data.
    plt.plot(pos[:,0],pos[:,1])
    plt.plot(a_1[:,0],a_1[:,1],'ro')
    plt.show()

    print('===================================')


##plt.plot(error_plot)
##plt.show()
    

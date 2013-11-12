import numpy as np
import matplotlib.pyplot as plt
import random


reg_order = 4
a_1 = np.asmatrix(np.loadtxt('./code/data/synthdata.csv',delimiter=','))
##print(a_1)

(N,M) = a_1.shape
pre_avg_MSE = 0

##print(N)

k_fold = 100 ## 100 is LOOCV, 10 is K10-CV

true_fit = np.zeros((reg_order+1,1))
for k in range(0,k_fold):
    for i in range(0,N/k_fold):
        hld = random.sample(xrange(N),N/k_fold)

    hld = np.sort(hld)
##    print(hld) 
    testD = []
    trainD = []
    hld_Count = 0
    hc = 0;

    
    for i in range(0,N):
        if i == hld[hld_Count]:
            ##print(hld_Count)
            testD = np.append(testD,a_1[hld[hld_Count]])
            if hld_Count < N/k_fold-1:
                hld_Count = hld_Count + 1
        else:
            trainD = np.append(trainD,a_1[i])
            
            



    testD = np.array(testD)
    trainD = np.array(trainD)
    testD.resize((N/k_fold),2)
    trainD.resize(N-(N/k_fold),2)

    (n,m) = trainD.shape
    ##print(n)

    B = np.array(trainD[:,1])
    B.resize((N-N/k_fold),1)
    ##print(B)
    ##wait = raw_input('')
    A_x = []
    ##print(A_x)

    for j in range(0,reg_order+1):
        A = np.array(trainD[0,0]**j)
        for i in range(1,n):
            A = np.append(A,trainD[i,0]**(j))
        ##print(A)
        A_x = np.append(A_x,A)

    A_x = np.array(A_x)
    A_x.resize((reg_order+1),(N-N/k_fold))
    At = A_x
    A_x = np.transpose(A_x)
##    print(A_x)
##    print(A_x.shape)

    AtA = np.dot(At,A_x)
    ##print(AtA)
    AtB = np.dot(At,B)
    invAtA = np.linalg.inv(AtA)
    fit = np.dot(invAtA,AtB)
    sumDeltaSq = 0
    for i in range(0,N/k_fold):
        y_hat = fit[0]
        for l in range(1,reg_order+1):
            y_hat = y_hat + fit[l]*(testD[i,0]**l)

        delta = y_hat - testD[i,1]
        delta_sq = delta**2
        sumDeltaSq = sumDeltaSq + delta_sq
    mSE = sumDeltaSq/(N/k_fold)
##    print('Mean Squared Error')
##    print(mSE)
    pre_avg_MSE = pre_avg_MSE + mSE
        
##    print(fit)
    true_fit = true_fit + fit

true_fit = true_fit/k_fold
print("TRUE FIT")
print(true_fit)
avg_MSE = pre_avg_MSE/(k_fold)
print(avg_MSE)
y=[]

x= np.arange(-5,6,1)
for i in range(-5,6):
    
    y_1 = true_fit[0]
    for l in range(1,reg_order+1):
        y_1 = y_1 + true_fit[l]*(i**l)
    y.append(y_1)

plt.plot(x,y)

plt.plot(a_1[:,0],a_1[:,1],'go')
plt.show()

plt.plot(x,y)




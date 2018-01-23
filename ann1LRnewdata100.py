import matplotlib.pyplot as plt
import numpy as np
#comment branch1
#comment branch2
# Load the data and create the data matrices X and Y
# This creates a feature vector X with a column of ones (bias)
# and a column of car weights.
# The target vector Y is a column of MPG values for each car.
#########################################################################
data=np.array(range(200))
datan=np.array(range(200))
datan100=np.array(range(100))
Y=np.array(range(100))

for i in range(0, 200):    
    data[i]=50*(1-np.exp(-i/20.))          
    datan[i]=45*(1-np.exp(-i/20.))+10*np.random.random() #noise 0..10
    #rads=2.*np.pi*i/200.
   # datan[i]=50*(1-np.exp(-i/20.))+10*np.random.random()*np.sin(4*rads)

plt.figure(1)  
#plt.plot(data)
plt.plot(datan)
plt.ylabel('weight signal')
plt.xlabel('time x100')

#########################################################################
#X_file = np.genfromtxt('mpg.csv', delimiter=',', skip_header=1)
#print np.shape(X_file) #X_File is 392x8) see csv
#N = np.shape(X_file)[0]
#print N #392
datan100=datan[100:200] #last 100 only where flat
#print np.size(datan100)
#plt.plot(datan100)
###########################################################################
#APPLY DIGITAL iir TO DATA 
#for i in range(1,100):#to remove spike at end
  # datan100[i]=0.3*datan100[i]+0.7*datan100[i-1] #less heavy filtering
# datan100[i]=0.11*datan100[i]+0.89*datan100[i-1]#FC=2HZ,FS=100HZ  
#datan100=datan100+5.0 #pulls down to 45 because of way waveform generated
#plt.plot(datan100) 
############################################################################
#N=200
N=100
#Xaxis=np.linspace(0,199,N)
Xaxis=np.linspace(0,99,N)
#print np.size(Xaxis), 
#print Xaxis

#X = np.hstack((np.ones(N).reshape(N, 1), X_file[:, 4].reshape(N, 1)))
X = np.hstack((np.ones(N).reshape(N, 1), Xaxis.reshape(N, 1)))
#print np.shape(X) #Xis 200x2
####X has leading 1 to uses as bias for neural
#Y = X_file[:, 0]
Y=datan100
#print np.size(Y)
#######################################################################
# Standardize the input 
#X[:, 1] = (X[:, 1]-np.mean(X[:, 1]))/np.std(X[:,1 )
Xaxisstd = (Xaxis[:]-np.mean(Xaxis[:]))/np.std(Xaxis[:])
#print np.size(Xaxisstd)

# There are two weights, the bias weight and the feature weight
w = np.array([1, 1])
#######################################################################
# Start batch gradient descent,
max_iter = 400 #100 not enough gives 43 use 400 gives 48 0r 49
eta = 1E-4 #step size
##### change this from 1E-3 to 1E-4 and it works 1E-5 takes longer to converge
########################################################################
for t in range(0, max_iter):
    # We need to iterate over each data point for one epoch
 
    grad_t = np.array([1., 1.])
    costfn=0
    for i in range(0, 99): #########################################
    
        #x_i = X[i, :]
        x_i = Xaxisstd[i]
        bias=1
        y_i = Y[i]        
       # h=w[0]*x_i[0]+w[1]*x_i[1]+w[2]*x_isq
        h=w[0]*bias+w[1]*x_i  
        err=h-y_i
        # print h          
        #grad_t[0] += 2*x_i[0]*err
        grad_t[0] += 2*bias*err
        #grad_t[1] += 2*x_i[1]*err
        grad_t[1] += 2*x_i*err 
        #easier to do this way than change x_i array above
        costfn=costfn + err*err
    #costfn= (1/2*N)( sum over N (err)^^2  
    # Update the weights
    w = w - eta*grad_t
    #print w
    costfn=costfn/(2*N)
   # print costfn #to check if grad descent working ok
print "Weights found:",w
##################################################################
#results are mostly within 1kg eg 2% but biassed on low side
##################################################################
# Plot the data and best fit line
plt.figure(2)
#tt = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
tt = np.linspace(np.min(Xaxisstd[:]), np.max(Xaxisstd[:]), 10)
#print tt #10 points only

bf_line = w[0]+w[1]*tt
#plt.plot(X[:, 1], Y, 'kx', tt, bf_line, 'r-')
plt.plot(Xaxisstd[:], Y,  tt, bf_line)
plt.ylim(0,60)
plt.xlabel('Time Normalised')
plt.ylabel('weight signal')
plt.title(' Regression on weight Data')

#plt.savefig('mpg.png')

plt.show()

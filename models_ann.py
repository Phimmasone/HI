import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
class annNET(object):
    '''Traditional neural networks with flexible/adjustable/customizable numbers of neurons and layers.'''
    def __init__(self,net_IHHO: list,initW: str,initB: str,*arg):
        self.description = "Neural Networks with flexible numbers of neurons and layers."
        np.random.seed(0)
        self.netPars,self.net_IHHO = {},net_IHHO
        self.initW,self.initB = initW,initB
        if initB.upper()=='ZERO': initB = 0 # Conditions to initial biases
        else: initB = 1
        print(">> Building neural network:")
        for l in range(1,len(net_IHHO)):
            w,b = 'w'+str(l)+str(l+1),'b'+str(l+1)
            if initW.upper()=='NORMAL':
                mu,var = 0,0.1
                self.netPars[w] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])
                self.netPars[b] = np.random.normal(mu,var,[1,net_IHHO[l]]) * initB
            elif initW.upper()=='GLOROT_UNIFORM':
                low = -(np.sqrt(6)/np.sqrt(net_IHHO[l-1]+net_IHHO[l]))
                hig = np.sqrt(6)/np.sqrt(net_IHHO[l-1]+net_IHHO[l])
                self.netPars[w] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]])
                self.netPars[b] = np.random.uniform(low,hig,[1,net_IHHO[l]]) * initB
            else:
                mu,var = 0,0.1
                self.netPars[w] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])
                self.netPars[b] = np.random.normal(mu,var,[1,net_IHHO[l]]) * initB
    def compile(self,alpha:float,cost:str,opti:str,acti:str,preStep:int,*arg):
        '''Compile function to assign or define hyper-parameters for network'''
        self.alpha,self.costName   = alpha,cost
        self.optName,self.actName  = opti,acti
        self.preStep = preStep
    # --- Update network's parameters ---
    def update_SGD(self,grad,*arg):  # 'SGD method'
        '''Stochatisitic gradient descent'''
        for key in self.netPars.keys(): # Loops through parameters
            self.netPars[key] -= self.alpha * grad[key].T
        return self.netPars
    def update_ADAM(self,t:int,grad,m0,v0,*arg): # 'ADAM method'
        ''' ADAM optimization is derived by paper of (Diederik P. Kingma, Jimmy Ba, 2014)'''
        beta_1,beta_2,epsilon = 0.9,0.999,1e-8
        t += 1
        mt,vt = {},{}
        for key in self.netPars.keys():  # Loops through parameters
            gt = grad[key].T.copy()
            mt[key] = beta_1*m0[key] + (1-beta_1) * gt
            vt[key] = beta_2*v0[key] + (1-beta_2) * (gt*gt)
            mt_hat = mt[key] / (1-beta_1**t)
            vt_hat = vt[key] / (1-beta_2**t)
            self.netPars[key] -= (self.alpha*mt_hat) / (np.sqrt(vt_hat)+epsilon)
        return self.netPars,mt,vt
    def update_RMSprop(self,grad,*arg):
        self.netPars = []
        return print('!!! Undefined function ...')
    def update_MOMENTUM(self,grad,*arg):
        self.netPars = []
        return print('!!! Undefined fucntion ...')
    # --- Optimize network ----
    def dropout(self,*arg):
        pars = {}
        return print('!!! Undefined function ...')
    # --- Cost function ---
    def cost(self,yPre,yTrue,costName:str):
        '''Cost/objective/error function'''
        if costName.upper()=='MSE': return np.sum((yPre - yTrue)**2)/np.max(yTrue.shape)
        elif costName.upper()=="LOGLOSS": return -(yTrue * np.log(yPre) + (1 - yTrue) * np.log(1 - yPre))
        else: KeyError('Cost function is not found !!!')
    def diffCost(self,yTrue,yPre,costName):
        '''Differentiated cost/objective/error functions'''
        if costName.upper()=='MSE': return yPre - yTrue
        else: KeyError('Differentiation of cost function is not found !!!')
    # --- Activation functions ---
    def neuron(self,s,actName:str,*arg):   # Activation functions
        '''Activate functions'''
        if actName.upper()=='SIGMOID': return 1/(1+np.exp(-s))
        elif actName.upper()=='TANH': return np.tanh(s)
        elif actName.upper()=='SOFTMAX': return np.exp(s)/np.sum(np.exp(s))
        elif actName.upper()=='RELU':
            zInd = np.where(s<=0)
            s[0,zInd[1]] = 0
            return s
        elif actName.upper()=='PRELU':
            zInd = np.where(s<0)
            s[0,zInd[1]] *= 0.001
            return s
        else:
            KeyError('Activation function is not found !!!')
    def difNeuron(self,s,actName:str,*arg):   # Differentiated actiation functions
        '''Differentiated activation functions'''
        if actName.upper()=='SIGMOID': return s * (1 - s)
        elif actName.upper()=='TANH': return 1 - np.tanh(s)**2
        elif actName.upper()=='SOFTMAX': return s*(1 - s)
        elif actName.upper()=='RELU':
            zInd,pInd = np.where(s<0),np.where(s>0)
            if len(zInd[1])>0: s[0,zInd[1]] = 0 
            if len(pInd[1])>0: s[0,pInd[1]] = 1
            return s
        elif actName.upper()=='PRELU':
            zInd,pInd = np.where(s<0),np.where(s>=0)
            if len(zInd[1])>0: s[0,zInd[1]] = 0.001
            if len(pInd[1])>0: s[0,pInd[1]] = 1
            return s
        else:
            KeyError('Differentiation of activation function is not found !!!')
    # --- Forward propagation ---
    def forward(self,x,*arg):
        '''Feed-forward propagations'''
        yHat = {}
        # --- Array must be always 2D ----
        if x.ndim == 1: xTr = np.expand_dims(x.copy(),0)
        elif x.ndim > 1: xTr = x.copy()
        # --- loops through entire data ---
        for l in range(1,len(self.net_IHHO)):
            w,b,y   = 'w'+str(l)+str(l+1),'b'+str(l+1),'y'+str(l)+str(l+1)
            yHat[y] = self.neuron(xTr.dot(self.netPars[w]) + self.netPars[b],self.actName)
            xTr = yHat[y].copy()
        return yHat[y],yHat
    # --- Backward propagation ---
    def backward(self,xIn,yTrue,yPre,*arg):
        '''Back-propagation algorithm'''
        # --- Array must be in 2D ---
        if xIn.ndim == 1: xIn = np.expand_dims(xIn,1)
        if yTrue.ndim == 1: yTrue= np.expand_dims(yTrue,1)
        grads,grad_act = {},{}  # arrays to store gradients
        keys = list(yPre.keys())
        keys.reverse()
        for i,key in enumerate(keys):
            w,b,dy = 'w' + key[1:],'b' + key[2],'dy' + key[2]
            if i==0:
                dyH = self.diffCost(yTrue,yPre[key])
                grad_act[dy] = dyH * self.difNeuron(yPre[key],self.actName)
                grads[w] = grad_act[dy].T.dot(yPre[keys[i+1]])
                grads[b] = grad_act[dy].T.dot(np.ones([len(grad_act[dy]),1])) 
            elif i>0 and i<len(keys)-1:
                dyPrev = 'dy' + keys[i-1][2]
                wPrev  = 'w' +  keys[i-1][1:]
                grad_act[dy] = grad_act[dyPrev].dot(self.netPars[wPrev].T) * self.difNeuron(yPre[keys[i]],self.actName)
                grads[w] = grad_act[dy].T.dot(yPre[keys[i+1]])
                grads[b] = grad_act[dy].T.dot(np.ones([len(grad_act[dy]),1]))  # times "1": dE/db = grad*1
            elif (i+1)==len(keys):
                dyPrev = 'dy' + keys[i-1][2]
                wPrev  = 'w' + keys[i-1][1:]
                grad_act[dy] = grad_act[dyPrev].dot(self.netPars[wPrev].T) * self.difNeuron(yPre[keys[i]],self.actName)
                grads[w] = grad_act[dy].T.dot(xIn)
                grads[b] = grad_act[dy].T.dot(np.ones([len(grad_act[dy]),1]))  
            else:
                KeyError('Computing gradients got an error !!!')
        return grads
    def predict(self,x,*arg):
        y_predict = np.zeros([x.shape[0],self.net_IHHO[-1]])
        for i in range(x.shape[0]):
            y_predict[i,:] = self.forward(x[i,:])
        return y_predict
    # --- Training network ---
    def train(self,dataTr,dataTs,epoch:int,batchSize:int,*arg):
        '''Training function'''
        # .... set up parameters ....
        x_train,y_train = dataTr  # output target or desired target must be always at the last column
        x_test,y_test   = dataTs
        outNode,model   = self.net_IHHO[-1],{}   # Get size of model and its output
        error_tr,error_ts = np.zeros(epoch),np.zeros(epoch) # Prepare empty arrays 
        tr_N,_ = x_train.shape   # Get samples of input for training 
        ts_N,_ = x_test.shape    # Get samples of input for validating
        # --- set up prediction step ahead ----
        if not self.preStep: self.preStep = 0 # self.preStep = Prediction step ahead
        # --- Pad zeros to xdata and xdata ---
        if (tr_N/batchSize)%1 != 0:
            N_padd1 = int(abs(np.ceil(tr_N/batchSize)*batchSize-tr_N))
            x_train = np.pad(x_train,((0,N_padd1),(0,0)),mode='constant',constant_values=0)
        if (ts_N/batchSize)%1 !=0:
            N_padd2 = int(abs(np.ceil(ts_N/batchSize)*batchSize-ts_N))
            x_test = np.pad(x_test,((0,N_padd2),(0,0)),mode='constant',constant_values=0)
        if y_test.shape[0]!=x_test.shape[0]:
            pad2 = abs(x_test.shape[0]-y_test.shape[0])
            y_test = np.pad(y_test,(0,pad2+self.preStep),'constant',constant_values=0)
        if y_train.shape[0]!=x_train.shape[0]:
            pad1 = abs(x_train.shape[0]-y_train.shape[0])
            y_train = np.pad(y_train,(0,pad1+self.preStep),'constant',constant_values=0)
        # --- expand 1D to 2D array ---- 
        if y_train.ndim==1: y_train = np.expand_dims(y_train,1)
        if y_test.ndim==1: y_test = np.expand_dims(y_test,1)
        # ... set up batchSize size ....
        if not batchSize: batchSize = 1
        else: batchSize = batchSize
        # .... Set up parameters for ADAM optimization ...
        if self.optName.upper()=='ADAM':
            m0,v0 = {},{}
            for key in self.netPars.keys(): # zero initialization
                m0[key],v0[key] = 0,0
        # ...... Training loops ........
        print(">> Training neural network: ")
        for i1 in range(0,epoch):  # Loops through self.epoch/iterations
            yHat_tr_rec  = np.zeros([y_train.shape[0],outNode])
            yHat_ts_rec  = np.zeros([y_test.shape[0],outNode])
            for i2 in range(0,x_train.shape[0],batchSize):  # Loops through training data samples
                xTr,yTr = x_train[i2:i2+batchSize,:].copy(),y_train[i2+self.preStep:i2+self.preStep+batchSize,:].copy()
                # .... Forward propagation ....
                yhat,yHat_tr = self.forward(xTr)
                yHat_tr_rec[i2:i2+batchSize,:] = yhat.copy()
                # .... Backward propagation .....
                grad = self.backward(xTr,yTr,yHat_tr)
                # .... Update parameters .....
                if self.optName.upper()=='SGD':
                    new_Pars = self.update_SGD(self.netPars,self.alpha,grad)
                elif self.optName.upper()=='ADAM':
                    new_Pars,mt,vt = self.update_ADAM(i2,grad,m0,v0)
                    m0,v0 = mt.copy(),vt.copy()
                # .... Assign new values to parameters ....
                self.netPars = new_Pars.copy()
            for i3 in range(0,x_test.shape[0],batchSize):  # Loops through validating data samples
                xTs = x_test[i3:i3+batchSize,:].copy()
                yHat_ts,_ = self.forward(xTs) # .... Forward propagation ....
                yHat_ts_rec[i3:i3+batchSize,:] = yHat_ts.copy()
            # .... Compute total errors ....
            error_tr[i1] = self.cost(yHat_tr_rec[:tr_N-self.preStep,:],y_train[self.preStep:tr_N,:])
            error_ts[i1] = self.cost(yHat_ts_rec[:ts_N-self.preStep,:],y_test[self.preStep:ts_N,:])
            print('>> Epoch: '+str(i1+1)+'/'+str(epoch)+'('+str(int(x_train.shape[0]/batchSize))+') || Train-'+self.costName.upper()+': '+str("{:.4f}".format(error_tr[i1]))+' - Test-'+self.costName.upper()+': '+str("{:.4f}".format(error_ts[i1])))
        # ..... Plot errors ......
        plt.figure()
        plt.plot(range(1,epoch+1),error_tr,'-'),plt.grid()
        plt.plot(range(1,epoch+1),error_ts,'-')
        plt.ylabel(str(self.costName.upper())),plt.xlabel('self.epoch')
        plt.legend(['Train','Test'],loc='upper right')
        # plt.ylim((0,1))
        plt.title("Model performance")
        plt.show()
        # ..... Assign outputs to model output .....
        model['tr_error'] = error_tr.copy()
        model['ts_error'] = error_ts.copy()
        model['netPars']  = self.netPars.copy()
        model['yTr_act']  = y_train[self.preStep:tr_N,:].copy()
        model['yTr_hat']  = yHat_tr_rec[:tr_N-self.preStep,:].copy()
        model['yTs_act']  = y_test[self.preStep:ts_N,:].copy()
        model['yTs_pre']  = yHat_ts_rec[:ts_N-self.preStep,:].copy()
        model['grads']    = grad.copy()
        model['net_IHHO'] = self.net_IHHO.copy()
        model['initW']    = self.initW
        model['initB']    = self.initB
        model['optName']  = self.optName
        model['forward']  = self.forward
        model['predict']  = self.predict
        model['NeuronFn'] = self.actName
        model['costName'] = self.costName
        return model
# /////////////////////////////////////////////////////////////////////////////////////
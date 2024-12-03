import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
class lstmNET(object):
    ''' This LSTM structure is mainly based on paper (Gers, Schmidhuber, and Cummins, 2000)'''
    def __init__(self,net_IHHO:list,batchSize:int,fback:int,Tsteps:int,initW:str,initB:str,*arg):
        self.description = "This lstmNET self.netPars is desinged with many-to-single types. Number of loopbacks, layers, and cells is customizable by user."
        np.random.seed(0) # set random values with same values for all the times
        self.netPars,self.net_IHHO = {},net_IHHO
        self.initW,self.initB = initW,initB
        if Tsteps<=1: Tsteps += 1 # --- Tsteps are always at least 2 as a minimum timestep ---
        if initB.upper()=="ZERO": Bmul = 0  # initial conditions of Bias
        else: Bmul = 1
        print(">> Building:")
        for l in range(1,len(net_IHHO)):
            wInd,celInd  = str(l)+str(l+1),str(l+1)
            if l>=1 and l<len(net_IHHO)-1:
                print('---> LSTM hidden layer '+str(l))
                h_l,c_l         = 'h'+celInd,'c'+celInd
                i_l,f_l,a_l,o_l = 'i'+celInd,'f'+celInd,'a'+celInd,'o'+celInd
                bi,bf,ba,bo,by  = 'bi'+celInd,'bf'+celInd,'ba'+celInd,'bo'+celInd,'by'+celInd
                Wxi,Wxf,Wxa,Wxo = 'Wxi'+wInd,'Wxf'+wInd,'Wxa'+wInd,'Wxo'+wInd
                Whi,Whf,Wha,Who = 'Whi'+wInd,'Whf'+wInd,'Wha'+wInd,'Who'+wInd
                # --- zero arrays: hidden and cell states, and gates ----
                self.netPars[h_l] = np.zeros([batchSize,Tsteps+1,net_IHHO[l]])
                self.netPars[c_l] = np.zeros([batchSize,Tsteps+1,net_IHHO[l]])
                self.netPars[i_l] = np.zeros([batchSize,Tsteps,net_IHHO[l]])
                self.netPars[f_l] = np.zeros([batchSize,Tsteps,net_IHHO[l]])
                self.netPars[a_l] = np.zeros([batchSize,Tsteps,net_IHHO[l]])
                self.netPars[o_l] = np.zeros([batchSize,Tsteps,net_IHHO[l]])
                if initW.upper()=="NORMAL":
                    mu,var = 0,0.1
                    # --- biases ---
                    self.netPars[bi]  = np.random.normal(mu,var,[1,net_IHHO[l]]) * Bmul
                    self.netPars[bf]  = np.random.normal(mu,var,[1,net_IHHO[l]]) * Bmul
                    self.netPars[ba]  = np.random.normal(mu,var,[1,net_IHHO[l]]) * Bmul
                    self.netPars[bo]  = np.random.normal(mu,var,[1,net_IHHO[l]]) * Bmul
                    # --- weights ---
                    self.netPars[Whi] = np.random.normal(mu,var,[net_IHHO[l],net_IHHO[l]])
                    self.netPars[Whf] = np.random.normal(mu,var,[net_IHHO[l],net_IHHO[l]])
                    self.netPars[Wha] = np.random.normal(mu,var,[net_IHHO[l],net_IHHO[l]])
                    self.netPars[Who] = np.random.normal(mu,var,[net_IHHO[l],net_IHHO[l]])
                    # --- weights ---
                    self.netPars[Wxi] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])
                    self.netPars[Wxf] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])
                    self.netPars[Wxa] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])
                    self.netPars[Wxo] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])   
                elif initW.upper()=="GLOROT_UNIFORM":
                    low = -(np.sqrt(6)/np.sqrt(net_IHHO[l-1]+net_IHHO[l]))
                    hig = np.sqrt(6)/np.sqrt(net_IHHO[l-1]+net_IHHO[l])
                    # --- biases ---
                    self.netPars[bi]  = np.random.uniform(low,hig,[1,net_IHHO[l]]) * Bmul
                    self.netPars[bf]  = np.random.uniform(low,hig,[1,net_IHHO[l]]) * Bmul
                    self.netPars[ba]  = np.random.uniform(low,hig,[1,net_IHHO[l]]) * Bmul
                    self.netPars[bo]  = np.random.uniform(low,hig,[1,net_IHHO[l]]) * Bmul
                    # --- weights ---
                    self.netPars[Whi] = np.random.uniform(low,hig,[net_IHHO[l],net_IHHO[l]])
                    self.netPars[Whf] = np.random.uniform(low,hig,[net_IHHO[l],net_IHHO[l]])
                    self.netPars[Wha] = np.random.uniform(low,hig,[net_IHHO[l],net_IHHO[l]])
                    self.netPars[Who] = np.random.uniform(low,hig,[net_IHHO[l],net_IHHO[l]])
                    # --- weights ---
                    self.netPars[Wxi] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]])
                    self.netPars[Wxf] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]])
                    self.netPars[Wxa] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]])
                    self.netPars[Wxo] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]]) 
            elif l==len(net_IHHO)-1:
                print('---> Output layer '+str(l))
                Why,by = 'Why'+wInd,'by'+celInd
                if initW.upper() == "NORMAL":
                    mu,var = 0,0.1
                    self.netPars[Why] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])
                    self.netPars[by]  = np.random.normal(mu,var,[1,net_IHHO[l]]) * Bmul
                elif initW.upper() == "GLOROT_UNIFORM":
                    low = -(np.sqrt(6)/np.sqrt(net_IHHO[l-1]+net_IHHO[l]))
                    hig = np.sqrt(6)/np.sqrt(net_IHHO[l-1]+net_IHHO[l])
                    self.netPars[Why] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]])
                    self.netPars[by]  = np.random.uniform(low,hig,[1,net_IHHO[l]]) * Bmul
            else: KeyError('>> LSTM layers are not correct: ' + str(l)+' is out of '+str(net_IHHO))
    # --- Activation functions ---
    def neuron(self,s,actName:str):   # Activation functions
        '''Activate functions'''
        if actName.upper()=='SIGMOID':
            return 1/(1+np.exp(-s))
        elif actName.upper()=='TANH':
            return np.tanh(s)
        elif actName.upper()=='SOFTMAX':
            return np.exp(self,s)/np.sum(np.exp(s))
        elif actName.upper()=='RELU':
            zInd = np.where(s<=0)
            s[0,zInd[1]] = 0
            return s
        elif actName.upper()=='PRELU':
            zInd = np.where(s<0)
            s[0,zInd[1]] *= 0.001
            return s
        elif actName.upper()=='GFN':
            return (4/(1+np.exp(-s)))-2
        elif actName.upper()=='HFN':
            return (2/(1+np.exp(-s)))-1
        else:
            KeyError('Activation function is not found !!!')
            return 0
    def difNeuron(self,s,actName:str):   # Differentiated actiation functions
        '''Differentiated activate functions'''
        if actName.upper()=='SIGMOID':
            return s*(1-s)
        elif actName.upper()=='GFN':
            return s*(4-s)
        elif actName.upper()=='HFN':
            return s*(2-s)
        elif actName.upper()=='TANH':
            return 1-np.tanh(s)**2
        elif actName.upper()=='SOFTMAX':
            return s*(1-s)
        elif actName.upper()=='RELU':
            zInd,pInd = np.where(self,s<0),np.where(s>0)
            if len(zInd[1])>0: s[0,zInd[1]] = 0 
            if len(pInd[1])>0: s[0,pInd[1]] = 1
            return s
        elif actName.upper()=='PRELU':
            zInd,pInd = np.where(self,s<0),np.where(s>=0)
            if len(zInd[1])>0: s[0,zInd[1]] = 0.001
            if len(pInd[1])>0: s[0,pInd[1]] = 1
            return s
        else:
            KeyError('The activate function is not found !!! as ...'+actName.upper)
    # --- Cost function ---
    def cost(self,yHat,yTru,actName:str):
        '''Cost/objective/error function'''
        if actName.upper()=='MSE': return np.sum(((yHat - yTru)**2))/len(yTru)
        else: KeyError('Cost function is not found !!!')
    def difCost(self,yTru,yHat,actName:str):
        '''differentiated cost/objective/error function'''
        if actName.upper()=='MSE': return yHat - yTru
        else:
            KeyError('Differentiation of cost function is not found !!!')
    def compile(self,alpha:float,Tsteps:int,costName:str,optName:str,actName:str,fback:int,outMode:str,preStep:int,*arg):
        '''This compile function is to define variables for model. '''
        self.alpha,self.Tsteps = alpha,Tsteps
        self.costName,self.optName,self.actName = costName,optName,actName
        self.fback,self.outMode,self.preStep = fback,outMode,preStep
    # ====- Forward propagation ====-
    def forward(self,x,*arg):
        ''' Forward propagation is baased on paper of ( Gers, Schmidhuber, and Cummins, 2000)'''
        # === Loops through layers ====
        for l in range(1,len(self.net_IHHO)):
            wInd,celInd = str(l)+str(l+1),str(l+1)
            if l < len(self.net_IHHO)-1: # from input layer to hidden layers
                h_l,c_l         = 'h'+celInd,'c'+celInd
                i_l,f_l,a_l,o_l = 'i'+celInd,'f'+celInd,'a'+celInd,'o'+celInd
                bi,bf,ba,bo,by  = 'bi'+celInd,'bf'+celInd,'ba'+celInd,'bo'+celInd,'by'+celInd
                Wxi,Wxf,Wxa,Wxo = 'Wxi'+wInd,'Wxf'+wInd,'Wxa'+wInd,'Wxo'+wInd
                Whi,Whf,Wha,Who = 'Whi'+wInd,'Whf'+wInd,'Wha'+wInd,'Who'+wInd
                # --- Case: there are more than one hidden layer ----
                if l==1: Xin = x.copy()  # at 1st hidden layer
                if l>1 and l<len(self.net_IHHO)-1: Xin = self.netPars[h_l].copy() # from 2nd hidden layer toward
                for t in range(0,self.Tsteps): # --- Loops through Tsteps --
                    # print('>> t-step: '+str(t))
                    h_stat = self.netPars[h_l][:,t,:].copy()
                    c_stat = self.netPars[c_l][:,t,:].copy()
                    Igate = self.neuron(Xin[:,t,:].dot(self.netPars[Wxi]) + h_stat.dot(self.netPars[Whi]) + self.netPars[bi],self.actName)
                    Fgate = self.neuron(Xin[:,t,:].dot(self.netPars[Wxf]) + h_stat.dot(self.netPars[Whf]) + self.netPars[bf],self.actName)
                    Agate = self.neuron(Xin[:,t,:].dot(self.netPars[Wxa]) + h_stat.dot(self.netPars[Wha]) + self.netPars[ba],'Tanh')
                    Ogate = self.neuron(Xin[:,t,:].dot(self.netPars[Wxo]) + h_stat.dot(self.netPars[Who]) + self.netPars[bo],self.actName)
                    c_stat = (Fgate * c_stat) + (Igate * Agate)
                    h_stat = Ogate * self.neuron(c_stat,'Tanh')
                    # ... Update and store states ...
                    self.netPars[h_l][:,t+1,:],self.netPars[c_l][:,t+1,:] = h_stat.copy(),c_stat.copy()
                    self.netPars[i_l][:,t,:],self.netPars[f_l][:,t,:] = Igate.copy(),Fgate.copy()
                    self.netPars[a_l][:,t,:],self.netPars[o_l][:,t,:] = Agate.copy(),Ogate.copy()
                    del Igate,Fgate,Ogate,Agate,h_stat,c_stat  # delete old parameters
                del Xin
            elif l == len(self.net_IHHO)-1: # at output layer
                Why,by,out_l,h_l = 'Why'+wInd,'by'+celInd,'out'+celInd,'h'+str(l)
                if self.outMode.upper()=="LAST":
                    Xin = self.netPars[h_l][:,-1,:].copy()
                Yout = self.neuron(Xin.dot(self.netPars[Why]) + self.netPars[by],self.actName)
                yHat = Yout.copy()
                self.netPars[out_l] = yHat.copy()
        return yHat
    # ==== Backward propagation ====
    def backward(self,x,yact,yHat,*arg):
        '''Back-propagation algorithms contain timesteps as back-propagation through time (BPTT)'''
        grads = {}  # array to store all gradients
        # === loops through layers ===
        for l in reversed(range(1,len(self.net_IHHO))):
            # --- Arrays to store summed up weights and biases ---
            dL_O,dL_Who,dL_Wxo,dL_bo = 1,0,0,0
            dL_A,dL_Wha,dL_Wxa,dL_ba = 1,0,0,0
            dL_F,dL_Whf,dL_Wxf,dL_bf = 1,0,0,0
            dL_I,dL_Whi,dL_Wxi,dL_bi = 1,0,0,0
            # === at output layer ===
            if l==len(self.net_IHHO)-1: # Calculate gradients at output layer
                Why,by,h_l,o_l,c_l = 'Why'+str(l)+str(l+1),'by'+str(l+1),'h'+str(l),'o'+str(l),'c'+str(l)
                dL_Why,dL_by,dL_h = 0,0,0
                # --- Cal gradients at output layer ----
                dL_y = self.difCost(yact,yHat,self.costName) * self.difNeuron(yHat,self.actName) # dL/dyhat
                if self.outMode.upper()=="LAST": 
                    h_stat = self.netPars[h_l][:,-1,:]  # output at last timestep
                # ... Gradients at output layer ....
                dL_Why += dL_y.T.dot(h_stat)
                dL_by  += dL_y.T.dot(np.ones([dL_y.shape[0],1]))
                # --- Gradient at output layer ----
                grads[Why],grads[by] = dL_Why.copy(),dL_by.copy() # store gradients
                # ---- Gradient w.r.t cell state and hidden state ====
                dL_h += dL_y.dot(self.netPars[Why].T)
                dL_C = dL_h*(self.netPars[o_l][:,-1,:] * self.difNeuron(self.netPars[c_l][:,-1,:],'Tanh'))
            # ==== at hidden layers ====
            elif l<len(self.net_IHHO)-1:  # calculate gradients at hidden layers backward
                wInd,celInd  = str(l)+str(l+1),str(l+1)
                # --- Index of weights and biases ---
                h_l,c_l         = 'h'+celInd,'c'+celInd
                i_l,f_l,a_l,o_l = 'i'+celInd,'f'+celInd,'a'+celInd,'o'+celInd
                bi,bf,ba,bo     = 'bi'+celInd,'bf'+celInd,'ba'+celInd,'bo'+celInd
                Wxi,Wxf,Wxa,Wxo = 'Wxi'+wInd,'Wxf'+wInd,'Wxa'+wInd,'Wxo'+wInd
                Whi,Whf,Wha,Who = 'Whi'+wInd,'Whf'+wInd,'Wha'+wInd,'Who'+wInd
                if l>1: # case: gradient at hidden layers
                    xTr = self.netPars['h'+str(l)].copy()
                if l==1: # case: gradients at input layer
                    xTr = x.copy()
                    h_l,c_l = 'h'+str(l+1),'c'+str(l+1)
                    i_l,f_l,a_l,o_l = 'i'+str(l+1),'f'+str(l+1),'a'+str(l+1),'o'+str(l+1)   
                # dL_C = dL_h*(self.netPars[o_l][:,-1,:] * self.difNeuron(self.netPars[c_l][:,-1,:],'Tanh'))
                grad_tt = {'dL_O':dL_h,'dL_A':dL_h,'dL_I':dL_h,'dL_F':dL_h,'dL_C':dL_C}  # to store gradients over timesteps
                for t in reversed(range(0,self.Tsteps)):  # --- loops through Tsteps backward ----
                    h_stat = self.netPars[h_l][:,t,:].copy()
                    c_stat = self.netPars[c_l][:,t,:].copy()
                    Igate  = self.netPars[i_l][:,t,:].copy()
                    Fgate  = self.netPars[f_l][:,t,:].copy()
                    Agate  = self.netPars[a_l][:,t,:].copy()
                    Ogate  = self.netPars[o_l][:,t,:].copy()
                    x_in   = xTr[:,t,:].copy()
                    # ... Gradients w.r.t h(t-1) ...
                    dL_O *= (grad_tt['dL_O'] * self.neuron(c_stat,'Tanh') * self.difNeuron(Ogate,self.actName))
                    dL_A *= (grad_tt['dL_A'] * (Ogate * self.difNeuron(c_stat,'Tanh')) * Igate * self.difNeuron(Agate,'Tanh'))
                    dL_I *= (grad_tt['dL_I'] * (Ogate * self.difNeuron(c_stat,'Tanh')) * Agate * self.difNeuron(Igate,self.actName))
                    dL_F *= (grad_tt['dL_F'] * (Ogate * self.difNeuron(c_stat,'Tanh')) * c_stat* self.difNeuron(Fgate,self.actName))
                    dL_F *= grad_tt['dL_C']
                    # ... Gradients at O gate ...
                    dL_Who += dL_O.T.dot(h_stat)
                    dL_Wxo += dL_O.T.dot(x_in)
                    dL_bo  += dL_O.T.dot(np.ones([dL_O.shape[0],1]))
                    # ... Gradients at A gate ...
                    dL_Wha += dL_A.T.dot(h_stat)
                    dL_Wxa += dL_A.T.dot(x_in)
                    dL_ba  += dL_A.T.dot(np.ones([dL_A.shape[0],1]))
                    # ... Gradients at I gate ...
                    dL_Whi += dL_I.T.dot(h_stat)
                    dL_Wxi += dL_I.T.dot(x_in)
                    dL_bi  += dL_I.T.dot(np.ones([dL_I.shape[0],1]))
                    # ... Gradients at F gate ...
                    dL_Whf += dL_F.T.dot(h_stat)
                    dL_Wxf += dL_F.T.dot(x_in)
                    dL_bf  += dL_F.T.dot(np.ones([dL_F.shape[0],1]))
                    # ... update ...
                    grad_tt['dL_O'] = dL_O.dot(self.netPars[Who].T)
                    grad_tt['dL_A'] = dL_A.dot(self.netPars[Wha].T)
                    grad_tt['dL_I'] = dL_I.dot(self.netPars[Whi].T)
                    grad_tt['dL_F'] = dL_F.dot(self.netPars[Whf].T)
                    grad_tt['dL_C'] += (grad_tt['dL_C'] * Fgate)
                # ... store gradients ...
                grads[Who],grads[Wxo],grads[bo] = dL_Who.copy(),dL_Wxo.copy(),dL_bo.copy()
                grads[Wha],grads[Wxa],grads[ba] = dL_Wha.copy(),dL_Wxa.copy(),dL_ba.copy()
                grads[Whf],grads[Wxf],grads[bf] = dL_Whf.copy(),dL_Wxf.copy(),dL_bf.copy()
                grads[Whi],grads[Wxi],grads[bi] = dL_Whi.copy(),dL_Wxi.copy(),dL_bi.copy()
                del Who,Whi,Whf,Wha,Wxo,Wxi,Wxf,Wxa,celInd,wInd,xTr,x_in,h_stat,c_stat,Igate,Fgate,Agate,Ogate  # clear old arrays
        return grads
    # --- Update self.netPars's parameters ---
    def update_SGD(self,grad,*arg):  # 'SGD method'
        '''Stochatistic gradient descents'''
        for key in grad.keys():   # Loops through parameters
            self.netPars[key] -= self.alpha * grad[key].T
        return self.netPars
    def update_ADAM(self,t:int,grads,m0,v0,*arg): # 'ADAM method'
        ''' ADAM optimization is derived by paper of (Diederik P. Kingma, Jimmy Ba, 2014)'''
        beta_1,beta_2,epsilon = 0.9,0.999,1e-8
        t += 1
        mt,vt = {},{}
        for key in grads.keys():   # Loops through parameters
            gt = grads[key].T.copy()
            mt[key] = beta_1*m0[key] + (1-beta_1) * gt
            vt[key] = beta_2*v0[key] + (1-beta_2) * (gt*gt)
            mt_hat = mt[key] / (1-beta_1**t)
            vt_hat = vt[key] / (1-beta_2**t)
            self.netPars[key] -= (self.alpha*mt_hat) / (np.sqrt(vt_hat)+epsilon)   
        return self.netPars,mt,vt
    # --- prediction ----
    def predict(self,x,*arg):
        '''Prediction is innerly based on right-side zero-padding. Zero-padded values are removed before return the outputs'''
        xIn,_ = self.zeroPad(x,[])
        y_predict = np.zeros([xIn.shape[0],self.net_IHHO[-1]])
        for i in range(0,xIn.shape[0],self.batchSize):
            y_predict[i:i+self.batchSize,:] = self.forward(xIn[i:i+self.batchSize,:,:])
        Nsam = xIn.shape[0]-x.shape[0]     # Number to ignore/remove zero-padded at the end
        y_predict = y_predict[:-Nsam,:] 
        return y_predict
    # ---- Zero padding ----
    def zeroPad(self,xdata,ydata,*arg):
        ''' Zero padding always pads zeros to the right-side of data'''
        N,_,_ = xdata.shape
        if (N/self.batchSize)%1 != 0:
            Npad1 = int(abs(np.ceil(N/self.batchSize)*self.batchSize-N))
            xdata_new = np.pad(xdata,((0,Npad1),(0,0),(0,0)),mode='constant',constant_values=0)
        else:
            xdata_new = xdata.copy()
        if len(ydata)!=0:
            if ydata.shape[0]!=xdata_new.shape[0]:
                pad2 = abs(xdata_new.shape[0]-ydata.shape[0])
                ydata_new = np.pad(ydata,((pad2,0),(0,0)),'constant',constant_values=0)
            if ydata_new.ndim==1: ydata_new = np.expand_dims(ydata_new,1)
        else:
            ydata_new = []
        return xdata_new,ydata_new
    # ==== Training ====-
    def train(self,dataTr,dataTs,epoch:int,batchSize:int,*arg):
        '''Training function'''
        # --- get hyper-parameters ---
        self.batchSize = batchSize
        net_model = {} # to store model and its components
        # ... Set up parameters for ADAM ...
        if self.optName.upper()=='ADAM':
            m0,v0 = {},{}
            for key in self.netPars.keys(): # zero initialization
                m0[key],v0[key] = 0,0
        # ... set up batch size ....
        if not batchSize: batchSize = 1 # default at 1
        else: batchSize = batchSize
        # --- Get Training and Testing data ----
        x_train,y_train = dataTr    # get X,Y data for training
        x_test,y_test   = dataTs    # get X,Y data for testing/validating
        tr_N,_,_ = x_train.shape
        ts_N,_,_ = x_test.shape
        # --- Pad zeros to xdata and xdata ---
        x_train,y_train = self.zeroPad(x_train,y_train)
        x_test,y_test   = self.zeroPad(x_test,y_test)
        # --- Arrays to store errors ----
        error_tr,error_ts = np.zeros([epoch,1]),np.zeros([epoch,1])
        # ======== Epoch ========
        print(">> Training LSTM network: ")
        for i1 in range(0,epoch):  # loops through epoches
            # ... Prepare arrays ...
            yTr_hat_rec = np.zeros([y_train.shape[0],self.net_IHHO[-1]])
            yHat_ts_rec = np.zeros([y_test.shape[0],self.net_IHHO[-1]])
            for i2 in range(0,x_train.shape[0],batchSize): # iteration through data series with Tsteps
                xTr = x_train[i2:i2+batchSize,:,:].copy()
                yTr = y_train[i2:i2+batchSize,:].copy()
                # === Forward ===
                yTr_hat = self.forward(xTr)
                # === Backward ===
                grads   = self.backward(xTr,yTr,yTr_hat)
                # === Update parameters ===
                if self.optName.upper()=='SGD':
                    new_Pars = self.update_SGD(grads)
                    self.netPars = new_Pars.copy()
                elif self.optName.upper()=='ADAM':
                    new_Pars,mt,vt = self.update_ADAM(i2,grads,m0,v0)
                    m0,v0 = mt.copy(),vt.copy()
                    self.netPars = new_Pars.copy()
                else:
                    KeyError('>> Optimizer is incorrect ...'+self.optName.upper())
                # ... Store actual & predicted values ...
                yTr_hat_rec[i2:i2+batchSize,:] = yTr_hat.copy()
            for i3 in range(0,x_test.shape[0],batchSize):  # validating/testing
                xTs = x_test[i3:i3+batchSize,:,:].copy()
                # ... make a validation or testing ...
                yHat_ts = self.forward(xTs)
                yHat_ts_rec[i3:i3+batchSize,:] = yHat_ts.copy()
            # ... Compute error or loss ...
            error_tr[i1,0] = self.cost(yTr_hat_rec[:tr_N-self.preStep,:],y_train[:tr_N-self.preStep,:],self.costName)
            error_ts[i1,0] = self.cost(yHat_ts_rec[:ts_N-self.preStep,:],y_test[:ts_N-self.preStep,:],self.costName)
            # ... show training errors ...
            print('>> Epoch: '+str(i1+1)+'/'+str(epoch)+'('+str(int(x_train.shape[0]/batchSize))+') || Train-'+self.costName.upper()+': '+str("{:.4f}".format(error_tr[i1,0]))+' -- Test-'+self.costName.upper()+': '+str("{:.4f}".format(error_ts[i1,0])))
        # ... Plot error ...
        plt.figure()
        plt.plot(error_tr,'-'),plt.grid(True)
        plt.plot(error_ts,'-')
        plt.legend(['Train','Test'],loc='upper right')
        plt.ylabel(self.costName.upper()),plt.xlabel('Epoch')
        # plt.ylim([0,1])
        plt.title('Model performance')
        plt.show()
        # ... Store info ...
        net_model['netPars']   = self.netPars.copy()
        net_model['initW']     = self.initW
        net_model['initB']     = self.initB
        net_model['grads']     = grads.copy()
        net_model['netStruct'] = self.net_IHHO.copy()
        net_model['yTr_act']   = y_train[:tr_N-self.preStep,:].copy()
        net_model['yTr_pre']   = yTr_hat_rec[:tr_N-self.preStep,:].copy()
        net_model['error_tr']  = error_tr.copy()
        net_model['error_ts']  = error_ts.copy()
        net_model['yTs_pre']   = yHat_ts_rec[:ts_N-self.preStep,:].copy()
        net_model['yTs_act']   = y_test[:ts_N-self.preStep,:].copy()
        net_model['forward']   = self.forward
        net_model['predict']   = self.predict
        net_model['loss']      = self.cost
        net_model['actName']   = self.actName

        return net_model
# /////////////////////////////////////////////////////////////////////////////////
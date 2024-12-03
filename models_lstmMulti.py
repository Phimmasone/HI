import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
'''
>> This code is developed by Phimmasone Thammavongsy Credit is required for using this open software.
Contact: phim.thmvsATgmail.com. National Uninversity of Laos (NUOL) and King Mongkut's Institute of Technology Ladkrbang (KMITL).
>> Use of this software still needs validations and corretions
'''
class lstmnetwork(object):
    def __init__(*arg):
        lstmnetwork.description = "This LSTM network is desinged with many-to-single types. Number of loopbacks, layers, and cells is customizable by user."
    def buildNet(net_IHHO,batchSize,fback,Tsteps,Winit,Binit,*arg):
        np.random.seed(0) # set random values with same values for all the times
        network = {}      # to store all network's parameters
        mu,var  = 0,0.1
        # --- Tsteps are always at least 2 as a minimum timestep ---
        if Tsteps<=1: Tsteps += 1
        if Binit.upper()=="ZERO": Bmul = 0
        else: Bmul = 1
        for l in range(1,len(net_IHHO)):
            wInd,celInd  = str(l)+str(l+1),str(l+1)
            if l>=1 and l<len(net_IHHO)-1:
                print('--- hidden layer '+str(l))
                h_l,c_l         = 'h'+celInd,'c'+celInd
                i_l,f_l,a_l,o_l = 'i'+celInd,'f'+celInd,'a'+celInd,'o'+celInd
                bi,bf,ba,bo,by  = 'bi'+celInd,'bf'+celInd,'ba'+celInd,'bo'+celInd,'by'+celInd
                Wxi,Wxf,Wxa,Wxo = 'Wxi'+wInd,'Wxf'+wInd,'Wxa'+wInd,'Wxo'+wInd
                Whi,Whf,Wha,Who = 'Whi'+wInd,'Whf'+wInd,'Wha'+wInd,'Who'+wInd
                # --- zero arrays: hidden and cell states, and gates ----
                network[h_l] = np.zeros([batchSize,Tsteps,net_IHHO[l]])
                network[c_l] = np.zeros([batchSize,Tsteps,net_IHHO[l]])
                network[i_l] = np.zeros([batchSize,Tsteps,net_IHHO[l]])
                network[f_l] = np.zeros([batchSize,Tsteps,net_IHHO[l]])
                network[a_l] = np.zeros([batchSize,Tsteps,net_IHHO[l]])
                network[o_l] = np.zeros([batchSize,Tsteps,net_IHHO[l]])
                if Winit.upper()=="NORMAL":
                    # --- biases ---
                    network[bi]  = np.random.normal(mu,var,[1,net_IHHO[l]]) * Bmul
                    network[bf]  = np.random.normal(mu,var,[1,net_IHHO[l]]) * Bmul
                    network[ba]  = np.random.normal(mu,var,[1,net_IHHO[l]]) * Bmul
                    network[bo]  = np.random.normal(mu,var,[1,net_IHHO[l]]) * Bmul
                    # --- weights ---
                    network[Whi] = np.random.normal(mu,var,[net_IHHO[l],net_IHHO[l]])
                    network[Whf] = np.random.normal(mu,var,[net_IHHO[l],net_IHHO[l]])
                    network[Wha] = np.random.normal(mu,var,[net_IHHO[l],net_IHHO[l]])
                    network[Who] = np.random.normal(mu,var,[net_IHHO[l],net_IHHO[l]])
                    # --- weights ---
                    network[Wxi] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])
                    network[Wxf] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])
                    network[Wxa] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])
                    network[Wxo] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])   
                elif Winit.upper()=="XAVIER":
                    low = -(np.sqrt(6)/np.sqrt(net_IHHO[l-1]+net_IHHO[l]))
                    hig = np.sqrt(6)/np.sqrt(net_IHHO[l-1]+net_IHHO[l])
                    # --- biases ---
                    network[bi]  = np.random.uniform(low,hig,[1,net_IHHO[l]]) * Bmul
                    network[bf]  = np.random.uniform(low,hig,[1,net_IHHO[l]]) * Bmul
                    network[ba]  = np.random.uniform(low,hig,[1,net_IHHO[l]]) * Bmul
                    network[bo]  = np.random.uniform(low,hig,[1,net_IHHO[l]]) * Bmul
                    # --- weights ---
                    network[Whi] = np.random.uniform(low,hig,[net_IHHO[l],net_IHHO[l]])
                    network[Whf] = np.random.uniform(low,hig,[net_IHHO[l],net_IHHO[l]])
                    network[Wha] = np.random.uniform(low,hig,[net_IHHO[l],net_IHHO[l]])
                    network[Who] = np.random.uniform(low,hig,[net_IHHO[l],net_IHHO[l]])
                    # --- weights ---
                    network[Wxi] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]])
                    network[Wxf] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]])
                    network[Wxa] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]])
                    network[Wxo] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]]) 
            elif l==len(net_IHHO)-1:
                print('--- Output layer '+str(l))
                Why,by = 'Why'+wInd,'by'+celInd
                if Winit.upper() == "NORMAL":
                    network[Why] = np.random.normal(mu,var,[net_IHHO[l-1],net_IHHO[l]])
                    network[by]  = np.random.normal(mu,var,[1,net_IHHO[l]]) * Bmul
                elif Winit.upper() == "XAVIER":
                    low = -(np.sqrt(6)/np.sqrt(net_IHHO[l-1]+net_IHHO[l]))
                    hig = np.sqrt(6)/np.sqrt(net_IHHO[l-1]+net_IHHO[l])
                    network[Why] = np.random.uniform(low,hig,[net_IHHO[l-1],net_IHHO[l]])
                    network[by]  = np.random.uniform(low,hig,[1,net_IHHO[l]]) * Bmul
            else:
                KeyError('>> Network layers are not correct: ' + str(l)+' is out of '+str(net_IHHO))
        return network
    # --- Activation functions ---
    def actFn(s,fnName):   # Activation functions
        if fnName.upper()=='SIGMOID':
            return 1/(1+np.exp(-s))
        elif fnName.upper()=='TANH':
            return np.tanh(s)
        elif fnName.upper()=='SOFTMAX':
            return np.exp(s)/np.sum(np.exp(s))
        elif fnName.upper()=='RELU':
            zInd = np.where(s<=0)
            s[0,zInd[1]] = 0
            return s
        elif fnName.upper()=='PRELU':
            zInd = np.where(s<0)
            s[0,zInd[1]] *= 0.001
            return s
        elif fnName.upper()=='GFN':
            return (4/(1+np.exp(-s)))-2
        elif fnName.upper()=='HFN':
            return (2/(1+np.exp(-s)))-1
        else:
            KeyError('Activation function is not found !!!')
            return 0
    def diffAct(s,fnName):   # Differentiated actiation functions
        if fnName.upper()=='SIGMOID':
            return s*(1-s)
        elif fnName.upper()=='GFN':
            return s*(4-s)
        elif fnName.upper()=='HFN':
            return s*(2-s)
        elif fnName.upper()=='TANH':
            return 1-np.tanh(s)**2
        elif fnName.upper()=='SOFTMAX':
            return s*(1-s)
        elif fnName.upper()=='RELU':
            zInd,pInd = np.where(s<0),np.where(s>0)
            if len(zInd[1])>0:
                s[0,zInd[1]] = 0 
            if len(pInd[1])>0:
                s[0,pInd[1]] = 1
            return s
        elif fnName.upper()=='PRELU':
            zInd,pInd = np.where(s<0),np.where(s>=0)
            if len(zInd[1])>0:
                s[0,zInd[1]] = 0.001
            if len(pInd[1])>0:
                s[0,pInd[1]] = 1
            return s
        else:
            KeyError('Differentiation of activation function is not found !!! as ...'+fnName.upper)
    # --- Cost function ---
    def costFn(yHat,yTru,fnName):
        if fnName.upper()=='MSE':
            return np.sum(((yHat - yTru)**2))/len(yTru)
        else:
            KeyError('Cost function is not found !!!')
            return 0
    def diffCost(yTru,yHat,fnName):
        if fnName.upper()=='MSE':
            return yHat - yTru
        else:
            KeyError('Differentiation of cost function is not found !!!')
            return 0
    # ------ Forwad propagation ------
    def forward(x,Tsteps,fback,netPars,lstm_IHHO,actName,outMode):
        for l in range(1,len(lstm_IHHO)):
            wInd,celInd = str(l)+str(l+1),str(l+1)
            if l < len(lstm_IHHO)-1:
                # print('----- LSTM layer '+str(l+1)+' --------')
                h_l,c_l         = 'h'+celInd,'c'+celInd
                i_l,f_l,a_l,o_l = 'i'+celInd,'f'+celInd,'a'+celInd,'o'+celInd
                bi,bf,ba,bo,by  = 'bi'+celInd,'bf'+celInd,'ba'+celInd,'bo'+celInd,'by'+celInd
                Wxi,Wxf,Wxa,Wxo = 'Wxi'+wInd,'Wxf'+wInd,'Wxa'+wInd,'Wxo'+wInd
                Whi,Whf,Wha,Who = 'Whi'+wInd,'Whf'+wInd,'Wha'+wInd,'Who'+wInd
                # --- Case: there are more than one hidden layer ----
                if l==1:
                    Xin_tr = x.copy()  # at 1st hidden layer
                if l>1 and l<len(lstm_IHHO)-1:
                    Xin_tr = netPars[h_l].copy() # from 2nd hidden layer and more
                # --- Loops through Tsteps ---
                for t in range(0,Tsteps):
                    # print('Cal: cell'+str(lstm_IHHO[l+1])+' at Tsteps = '+str(t))
                    h_prev = netPars[h_l][:,t-1,:].copy()
                    c_prev = netPars[c_l][:,t-1,:].copy()
                    Igate = lstmnetwork.actFn(Xin_tr[:,t,:].dot(netPars[Wxi]) + h_prev.dot(netPars[Whi]) + netPars[bi],actName)
                    Fgate = lstmnetwork.actFn(Xin_tr[:,t,:].dot(netPars[Wxf]) + h_prev.dot(netPars[Whf]) + netPars[bf],actName)
                    Agate = lstmnetwork.actFn(Xin_tr[:,t,:].dot(netPars[Wxa]) + h_prev.dot(netPars[Wha]) + netPars[ba],actName)
                    Ogate = lstmnetwork.actFn(Xin_tr[:,t,:].dot(netPars[Wxo]) + h_prev.dot(netPars[Who]) + netPars[bo],'Tanh')
                    Cstate = Fgate * c_prev + Igate * Agate
                    Hstate = Ogate * lstmnetwork.actFn(Cstate,'Tanh')
                    # ... Update and store states ...
                    netPars[h_l][:,t,:],netPars[c_l][:,t,:] = Hstate.copy(),Cstate.copy()
                    netPars[i_l][:,t,:],netPars[f_l][:,t,:] = Igate.copy(),Fgate.copy()
                    netPars[a_l][:,t,:],netPars[o_l][:,t,:] = Agate.copy(),Ogate.copy()
                    del Igate,Fgate,Ogate,Agate,Hstate,Cstate  # delete old parameters
                del Xin_tr
            elif l == len(lstm_IHHO)-1:
                Why,by,out_l,h_l = 'Why'+wInd,'by'+celInd,'out'+celInd,'h'+str(l)
                # print('----- Dense layer '+str(l+1)+' --------')
                if outMode.upper()=="LAST":
                    Y = lstmnetwork.actFn(netPars[h_l][:,-1,:].dot(netPars[Why]) + netPars[by],actName)
                    yHat = Y.copy()
                    netPars[out_l] = yHat.copy()
        return yHat,netPars
    # ----- Backward propagation -----
    def backward(x,yact,yHat,Tsteps,netPars,lstm_IHHO,costName,actiName,outMode):
        grads = {}  # array to store all gradients
        if yact.ndim==1:
            yact = np.expand_dims(yact,1)
        # --- loops through layers ----
        for l in reversed(range(1,len(lstm_IHHO))):
            wInd,celInd  = str(l)+str(l+1),str(l+1)
            # --- Index of weights and biases ---
            h_l,c_l         = 'h'+celInd,'c'+celInd
            i_l,f_l,a_l,o_l = 'i'+celInd,'f'+celInd,'a'+celInd,'o'+celInd
            bi,bf,ba,bo     = 'bi'+celInd,'bf'+celInd,'ba'+celInd,'bo'+celInd
            Wxi,Wxf,Wxa,Wxo = 'Wxi'+wInd,'Wxf'+wInd,'Wxa'+wInd,'Wxo'+wInd
            Whi,Whf,Wha,Who = 'Whi'+wInd,'Whf'+wInd,'Wha'+wInd,'Who'+wInd
            # --- Arrays to store summed up weights and biases ---
            dL_C,dL_Cprev = 0,0
            dL_O,dL_Who,dL_Wxo,dL_bo = 0,0,0,0
            dL_A,dL_Wha,dL_Wxa,dL_ba = 0,0,0,0
            dL_F,dL_Whf,dL_Wxf,dL_bf = 0,0,0,0
            dL_I,dL_Whi,dL_Wxi,dL_bi = 0,0,0,0
            # ---- at output layer ----
            if l==len(lstm_IHHO)-1: # Calculate gradients at output layer
                dL_Why,dL_by = 0,0
                Why,by,h_l = 'Why'+str(l)+str(l+1),'by'+str(l+1),'h'+str(l)
                # --- Cal gradients at output layer ----
                dL_y = lstmnetwork.diffCost(yact,yHat,costName) * lstmnetwork.diffAct(yHat,actiName) # dL/dyhat
                if outMode.upper()=="LAST": 
                    h_prev = netPars[h_l][:,-1,:]  # output at last timestep
                # ... Gradients at output layer ....
                dL_Why += dL_y.T.dot(h_prev)
                dL_by  += dL_y.T.dot(np.ones([dL_y.shape[0],1]))
                # --- Gradient at hidden state ----
                dL_h = (dL_y * lstmnetwork.diffAct(yHat,actiName)).dot(netPars[Why].T)
                grads[Why],grads[by] = dL_Why.copy(),dL_by.copy() # store gradients
            # ---- at hidden layers -----
            elif l<len(lstm_IHHO)-1:  # calculate gradients at hidden layers backward
                if l>1: # case: gradient at hidden layers
                    Xin_tr = netPars['h'+str(l)].copy()
                if l==1: # case: gradients at input layer
                    Xin_tr = x.copy()
                    h_l,c_l = 'h'+str(l+1),'c'+str(l+1)
                    i_l,f_l,a_l,o_l = 'i'+str(l+1),'f'+str(l+1),'a'+str(l+1),'o'+str(l+1)   
                # --- loops through Tsteps backward ---- 
                for t_rev in reversed(range(0,Tsteps)):
                    # print('Cal gradients at Tsteps '+str(t_rev))
                    h_prev = netPars[h_l][:,t_rev,:].copy()
                    c_prev = netPars[c_l][:,t_rev,:].copy()
                    Igate = netPars[i_l][:,t_rev,:].copy()
                    Fgate = netPars[f_l][:,t_rev,:].copy()
                    Agate = netPars[a_l][:,t_rev,:].copy()
                    Ogate = netPars[o_l][:,t_rev,:].copy()
                    # ... Gradients at states and gates ...
                    # dL_h   += (dL_y * lstmnetwork.diffAct(yHat,actiName)).dot(netPars[Why].T)
                    dL_C   += (dL_h * Ogate * lstmnetwork.diffAct(c_prev,'Tanh'))
                    dL_Cprev+= (dL_C * Fgate)
                    dL_C   += dL_Cprev
                    dL_O   += (dL_h * lstmnetwork.actFn(c_prev,'Tanh'))
                    dL_F   += (dL_C * c_prev)
                    dL_I   += (dL_C * Agate)
                    dL_A   += (dL_C * Igate)
                    # ... Gradients at O gate ...
                    dL_Who += (dL_O * lstmnetwork.diffAct(Ogate,'Tanh')).T.dot(h_prev)
                    dL_Wxo += (dL_O * lstmnetwork.diffAct(Ogate,'Tanh')).T.dot(Xin_tr[:,t_rev,:])
                    dL_bo  += (dL_O * lstmnetwork.diffAct(Ogate,'Tanh')).T.dot(np.ones([dL_O.shape[0],1]))
                    # ... Gradients at F gate ...
                    dL_Whf += (dL_F * lstmnetwork.diffAct(Fgate,actiName)).T.dot(h_prev)
                    dL_Wxf += (dL_F * lstmnetwork.diffAct(Fgate,actiName)).T.dot(Xin_tr[:,t_rev,:])
                    dL_bf  += (dL_F * lstmnetwork.diffAct(Fgate,actiName)).T.dot(np.ones([dL_F.shape[0],1]))
                    # ... Gradients at I gate ...
                    dL_Whi += (dL_I * lstmnetwork.diffAct(Igate,actiName)).T.dot(h_prev)
                    dL_Wxi += (dL_I * lstmnetwork.diffAct(Igate,actiName)).T.dot(Xin_tr[:,t_rev,:])
                    dL_bi  += (dL_I * lstmnetwork.diffAct(Igate,actiName)).T.dot(np.ones([dL_I.shape[0],1]))
                    # ... Gradients at A gate ...
                    dL_Wha += (dL_A * lstmnetwork.diffAct(Agate,actiName)).T.dot(h_prev)
                    dL_Wxa += (dL_A * lstmnetwork.diffAct(Agate,actiName)).T.dot(Xin_tr[:,t_rev,:])
                    dL_ba  += (dL_A * lstmnetwork.diffAct(Agate,actiName)).T.dot(np.ones([dL_A.shape[0],1]))
                # ... store gradients ...
                grads[Who],grads[Wxo],grads[bo] = dL_Who.copy(),dL_Wxo.copy(),dL_bo.copy()
                grads[Wha],grads[Wxa],grads[ba] = dL_Wha.copy(),dL_Wxa.copy(),dL_ba.copy()
                grads[Whf],grads[Wxf],grads[bf] = dL_Whf.copy(),dL_Wxf.copy(),dL_bf.copy()
                grads[Whi],grads[Wxi],grads[bi] = dL_Whi.copy(),dL_Wxi.copy(),dL_bi.copy()
                del Who,Whi,Whf,Wha,Wxo,Wxi,Wxf,Wxa,celInd,wInd,Xin_tr,h_prev,c_prev,Igate,Fgate,Agate,Ogate  # clear old arrays
        return grads
    # --- Update network's parameters ---
    def update_SGD(netPars,alpha,grad):  # 'SGD method'
        for key in grad.keys(): # Loops through parameters
            netPars[key] -= alpha * grad[key].T
        return netPars
    def update_ADAM(t,netPars,alpha,grads,adamPars,m0,v0): # 'ADAM method'
        beta_1,beta_2,epsilon = adamPars.copy()
        t += 1
        mt,vt = {},{}
        for key in grads.keys():  # Loops through parameters
            gt = grads[key].T.copy()
            mt[key] = beta_1*m0[key] + (1-beta_1) * gt
            vt[key] = beta_2*v0[key] + (1-beta_2) * (gt*gt)
            mt_hat = mt[key] / (1-beta_1**t)
            vt_hat = vt[key] / (1-beta_2**t)
            netPars[key] -= (alpha*mt_hat) / (np.sqrt(vt_hat)+epsilon)   
        return netPars,mt,vt
    # ----- Training ------
    def train(dataTr,dataTs,lstmPars,hyperPars):
        # --- get hyper-parameters ---
        epoch,alpha,Tsteps,costName,optiName,actiName,weight_init,bias_init,batch,fback,lstm_IHHO,outMode,preStep = hyperPars
        net_model = {} # to store model and its components
        # ... Set up parameters for ADAM ...
        if optiName.upper()=='ADAM':
            beta_1,beta_2,epsilon = 0.9,0.999,1e-8
            m0,v0 = {},{}
            for key in lstmPars.keys(): # zero initialization
                m0[key],v0[key] = 0,0
            adamPars = [beta_1,beta_2,epsilon]
        # ... set up batch size ....
        if not batch: batchSize = 1 # default at 1
        else: batchSize = batch
        # --- Tsteps are always at least 2 ----
        if Tsteps<=1: Tsteps += 1
        # --- Get Training and Testing data ----
        xdataTr,ydataTr = dataTr    # get X,Y data for training
        xdataTs,ydataTs = dataTs    # get X,Y data for testing/validating
        dataN1,_,_ = xdataTr.shape
        dataN2,_,_ = xdataTs.shape
        # --- Pad zeros to ydata based on xdata length ---
        if ydataTs.shape[0]!=dataN2:
            ydataTs = np.pad(ydataTs,(0,abs(dataN2-ydataTs.shape[0])),'constant')
        if ydataTr.shape[0]!=dataN1:
            ydataTr = np.pad(ydataTr,(0,abs(dataN1-ydataTr.shape[0])),'constant')
        # --- Expand 1D to 2D data ----- 
        if ydataTr.ndim==1: ydataTr = np.expand_dims(ydataTr,1)
        if ydataTs.ndim==1: ydataTs = np.expand_dims(ydataTs,1)
        # --- Arrays to store errors ----
        error_tr,error_ts = np.zeros([epoch,1]),np.zeros([epoch,1])
        for i1 in range(0,epoch):  # loops through epoches
            # ... Prepare arrays ...
            yHat_tr_rec = np.zeros([dataN1,lstm_IHHO[-1]])
            yHat_ts_rec  = np.zeros([dataN2,lstm_IHHO[-1]])
            for i2 in range(0,dataN1-batchSize,batchSize): # iteration through data series with Tsteps
                xIn_tr = xdataTr[i2:i2+batchSize,:,:].copy()
                yIn_tr = ydataTr[i2+preStep:i2+preStep+batchSize,:].copy()
                # ... Forward ...
                yHat_tr,lstmPars = lstmnetwork.forward(xIn_tr,Tsteps,fback,lstmPars,lstm_IHHO,actiName,outMode)
                # ... Backward ...
                grads = lstmnetwork.backward(xIn_tr,yIn_tr,yHat_tr,Tsteps,lstmPars,lstm_IHHO,costName,actiName,outMode)
                # ... Update parameters ...
                if optiName.upper()=='SGD':
                    new_netPars = lstmnetwork.update_SGD(lstmPars,alpha,grads)
                    lstmPars = new_netPars.copy()
                elif optiName.upper()=='ADAM':
                    new_netPars,mt,vt = lstmnetwork.update_ADAM(i2,lstmPars,alpha,grads,adamPars,m0,v0)
                    m0,v0 = mt.copy(),vt.copy()
                    lstmPars = new_netPars.copy()
                else:
                    KeyError('>> Optimizer is incorrect ...'+optiName.upper())
                # ... Store actual & predicted values ...
                yHat_tr_rec[i2:i2+batchSize,:] = yHat_tr.copy()
            for i3 in range(0,dataN2-batchSize,batchSize):  # validating/testing
                Xin_ts = xdataTs[i3:i3+batchSize,:,:].copy()
                # ... make a validation or testing ...
                yHat_ts,_ = lstmnetwork.forward(Xin_ts,Tsteps,fback,lstmPars,lstm_IHHO,actiName,outMode)
                yHat_ts_rec[i3:i3+batchSize,:] = yHat_ts.copy()
            # ... Compute error or loss ...
            error_tr[i1,0] = lstmnetwork.costFn(yHat_tr_rec,ydataTr,costName)
            error_ts[i1,0] = lstmnetwork.costFn(yHat_ts_rec,ydataTs,costName)
            # ... show training errors ...
            print('>> Epoch: '+str(i1+1)+'/'+str(epoch)+' ||  Train-'+costName.upper()+': '+str(error_tr[i1,0])+' --- Validate-'+costName.upper()+':'+str(error_ts[i1,0]))
        # ... Plot error ...
        plt.figure()
        plt.plot(error_tr,'-'),plt.grid(True)
        plt.plot(error_ts,'-')
        plt.legend(['Train','Test'])
        plt.ylabel(costName.upper()),plt.xlabel('Epoch')
        # plt.ylim([0,np.max(error_tr)])
        plt.show()
        # ... Store info ...
        net_model['lstm_net']  = lstmPars.copy()
        net_model['initW']     = weight_init
        net_model['initB']     = bias_init
        net_model['gradients'] = grads.copy()
        net_model['netPars']   = lstmPars.copy()
        net_model['yAct_tr']   = ydataTr.copy()
        net_model['yHat_tr']   = yHat_tr_rec.copy()
        net_model['error_tr']  = error_tr.copy()
        net_model['error_ts']  = error_ts.copy()
        net_model['yVal_pre']  = yHat_ts_rec.copy()
        net_model['yVal_act']  = ydataTs.copy()
        net_model['predict']   = lstmnetwork.forward
        net_model['loss_fn']   = lstmnetwork.costFn
        return net_model
# /////////////////////////////////////////////////////////////////////////////////
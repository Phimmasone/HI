import numpy as np
class process(object):
    '''Functions to process data as requirements.'''
    def __init__(*arg):
        process.description = "Data proccessing techniques"
    def splitData(x,y,ratio,shuffle: bool,*arg):
        '''Data split function to split data into sets following given ratio'''
        if shuffle: # shuffle data position
            trRat,_ = ratio
            dataTr = int((trRat * len(y))/100)
            dataRan = np.arange(0,len(y),1)
            np.random.shuffle(dataRan)
            TrInd = dataRan[:dataTr]
            TsInd = dataRan[dataTr+1:]
            xtr,xts = x[TrInd,:],x[TsInd,:]
            ytr,yts = y[TrInd],y[TsInd]
        elif not shuffle:
            trRat,_ = ratio
            dataTr  = int((trRat * len(y))/100)
            TrInd = np.arange(0,dataTr)
            TsInd = np.arange(dataTr+1,len(y))
            xtr,xts = x[TrInd,:],x[TsInd,:]
            ytr,yts = y[TrInd],y[TsInd]
        else:
            KeyError('>> Spliting method is not found ...')
        return xtr,ytr,xts,yts
    def scale(x,meth:str):
        '''Scale data signal functions, Min-max normalization and Standardization method.'''
        if meth.upper()=='MINMAX':
            xnew = (x - x.min(0))/(x.max(0)-x.min(0))
            if xnew.ndim==1: xnew = np.expand_dims(xnew,1)
            return xnew
        elif meth.upper()=='STD':
            xnew = (x - x.mean(0))/np.sqrt(np.sum((x-x.mean(0))**2)/len(x))
            if xnew.ndim==1: xnew = np.expand_dims(xnew,1)
            return xnew
    def unscale(x,minmax,meth:str):
        '''Unscale function of the min-max normalization method, you need to collect minimum and maximum values of your target'''
        if meth.upper()=='MINMAX':
            xmin,xmax = minmax  # min and max values of your original data
            return (x*(xmax-xmin))+xmin
    def mavgW(x,w:int,*arg): # (data,windowSize)
        ''' Running moving average of X data based on W window size. '''
        out = np.zeros_like(x)
        for i in range(len(x)-w):
            out[i] = np.mean(x[i:i+w])
        out[-w:] = out[-w-1].copy()
        return out
    def removeJump(x,method:str):
        '''Remove single jumped value in array. For >1 jumped values, you need to reuse it respecitively'''
        if method.upper()=="UP":
            Val_ind = (x == np.max(x))
        elif method.upper()=="DOWN":
            Val_ind = (x == np.min(x))
        else:
            KeyError('>> Given name of method does not exist as ... '+method)
        max_ind = np.where(Val_ind==True)[0]
        if len(max_ind) > 1:
            left,right = max_ind[0]-1, max_ind[-1]+1
        elif len(max_ind) == 1:
            left,right = max_ind-1,max_ind+1 
        xnew = x.copy()
        xnew[max_ind] = np.mean([xnew[left],xnew[right]]).copy()
        return xnew
    def reshape2Dto3D(data2D,tsteps:int,*arg):
        ''' Reshape input dataset from 2D matrix [sequences,features] to 3D matrix [sequences,timesteps,features]'''
        N,M = data2D.shape
        if tsteps<=1: tsteps +=1       #  tstepss are always at least 2
        data3D = np.zeros([N,tsteps,M])
        data2D = np.pad(data2D,((0,tsteps),(0,0)),mode='constant',constant_values=0)
        for i in range(0,N,1):
            data3D[i,:,:] = data2D[i:i+tsteps,:].copy()
        return data3D
    def shiftTarget(target,prStep:int,*arg):
        ''' prStep is to shift target/label/desired output ahead for training model. Thus, model utilization is also based on this condition.'''
        if target.ndim==1:
            N = len(target)
            target_new = target[prStep:].copy()
            pad1 = N -  len(target_new)
            target_new = np.pad(target_new,(0,pad1),mode='constant',constant_values=0)

        elif target.ndim>1 and target.shape[0]>target.shape[1]:
            N,_ = target.shape
            target_new = target[prStep:,:].copy()
            pad2 = N - target_new.shape[0]
            target_new = np.pad(target_new,((0,pad2),(0,0)),mode='constant',constant_values=0)
        if target_new.ndim==1:
            target_new = np.expand_dims(target_new,1)

        return target_new
# da = np.arange(0,3*20).reshape([20,3])
# data = process.conv2Dto3D(da,5,3)
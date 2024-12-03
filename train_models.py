from functions.models_ann import annNET # type: ignore
from functions.models_lstm import lstmNET # type: ignore
from functions.data_process import process # type: ignore
from functions import datasetKMITL as data # type: ignore
from functions.plotResults import plots # type: ignore
from functions.predicts import predicts # type: ignore
import matplotlib.pyplot as plt # type: ignore
from datetime import datetime # type: ignore
import sklearn.datasets as datasets # type: ignore
import numpy as np # type: ignore
today = datetime.today() 
# ---- Get data and input feature selection ------
# data 01: 0.HrS, 1.HrC, 2.DnS, 3.DnC, 4.kp8, 5.Kp, 6.ap8, 7.Ap, 8.SSN, 9.F10.7, 10.F10.7adj, 11.Vz, 12.GW, 13.h'F, 14.h'Fpss, 15.Vzpss
features  = [0,1,2,3,6,10,14,15,12]
# data 02: 0.HrS, 1.HrC, 2.DnS, 3.DnC, 4.kp8, 5.ap8, 6.SSN, 7.F10.7adj, 8.hF, 9.Vz, 10.hftpss, 11.Vztpss, 12.GWtpss
features1 = [0,1,2,3,5,7,10,11,12]
xdata,ydata = data.datasets.dataV2()
# --- Ap conditions ---- 
ApD = xdata[:,7].copy()
Apls16_ind = ApD<=16    # get index 
Apgt16_ind = ApD>16     # get index
# --- Process/clean data F10.7 by removing jumped values ----
xdata[:,10] = process.removeJump(xdata[:,10],"UP")
xdata[:,9]  = process.removeJump(xdata[:,9],"UP")
xdata[:,6]  = process.removeJump(xdata[:,6],"UP")
xdata[:,7]  = process.removeJump(xdata[:,7],"UP")
xdata[:,13] = process.removeJump(xdata[:,13],"DOWN")
xdata[:,14] = process.removeJump(xdata[:,14],"DOWN")
# --- calculate moving average -----
apavg2   = process.mavgW(xdata[:,6],2*12)
apavg4   = process.mavgW(xdata[:,6],4*12)
apavg8  = process.mavgW(xdata[:,6],8*12)
f10avg27 = process.mavgW(xdata[:,10],27*52)
f10avg81 = process.mavgW(xdata[:,10],81*52)
f10P     = (xdata[:,10]+f10avg81)/2
SNavg27  = process.mavgW(xdata[:,8],27*52)
SNavg81  = process.mavgW(xdata[:,8],81*52)
# --- get data by feature selection ----
xdata = xdata[:,features]
# --- change data features ----
xdata[:,2] = apavg8.copy()
xdata[:,3] = f10P.copy()
# --- select data by Ap condition ----
xdata = xdata[Apls16_ind,:].copy()
ydata = ydata[Apls16_ind].copy()
## ====== datasets from sklearn libraries =====
# data  = datasets.load_breast_cancer()
# xdata = data['data'].copy()
# ydata = data['target'].copy()
# ---- data normalization ----
xdata = process.scale(xdata,"minmax")
ydata = process.scale(ydata,"minmax")
# --- Ignore NaN in data to become zero ----
x_zero,y_zero = np.isnan(xdata),np.isnan(ydata)
xdata[x_zero],ydata[y_zero] = 0,0
# ---- Split data into training and testing/validating sets -----
xtr,ytr,xts,yts = process.splitData(xdata,ydata,[90,10],shuffle=False) # (dataset,[train_ratio,test_ratio],"method")
# ---- Plot inputs ----
# plots.plotInputs(xtr)
# plots.plotInputs(xts)
# ====== Define network structure =======
_,xIn    = xtr.shape          # get number of inputs: [sequences,features]
net_IHHO = [xIn,120,1]        # [Input layer, [hidden layers], output layer]
# ======= Hyper-parameters ========
alpha     = 1e-3              # learning rate
epoch     = 20               # training iterations
cost      = 'mse'             # objective/cost/error function
opti      = 'adam'            # optimization method: 'sgd','adam'
acti      = 'sigmoid'         # activation function: 'SIGMOID','TANH','RELU
Winit     = 'glorot_uniform'  # weight initialization method: 'normal','glorot_uniform','[]'
Binit     = 'glorot_uniform'  # bias initialization method: 'zero','glorot_uniform','normal','[]'
batchSize = 128               # batch size of each update parameters
preStep   = 52                # step of prediction ahead, "0" means to at time t and otherwise, ahead step
# ===== Parameters of LSTM network only =======
outMode   = 'last'            # select output: many2one('last')
Tsteps    = 8                 # Tsteps of input series for LSTM cells ('1' means to [t-1, t])
fback     = 0                 # Output feeds to the next layer: NOTE: it does not support this option yet
# ========  Artificial Neurnal Networks =========
# dataTr,dataTs = [xtr,ytr],[xts,yts]
# np.save('xTrESF_shiftANN',xtr)
# np.save('yTrESF_shiftANN',ytr)
# np.save('xTsESF_shiftANN',xts)
# np.save('yTsESF_shiftANN',yts)
## --- Build and train network: ANN ----
# ANN  = annNET(net_IHHO,Winit,Binit)
# ANN.compile(alpha,cost,opti,acti,preStep)
# model= ANN.train(dataTr,dataTs,epoch,batchSize)
# --- export/save model and datasets ----
# np.save("ESFANN_Model"+str(today)[:10],model) # save model  # --- save model
# ======== Long-Short Term Memory  ==========
xtr3D = process.reshape2Dto3D(xtr,Tsteps)   # Tsteps or timesteps to reshape x data for training 
xts3D = process.reshape2Dto3D(xts,Tsteps)
ytr   = process.shiftTarget(ytr,preStep)    # preStep to shift target ahead
yts   = process.shiftTarget(yts,preStep)
dataTr,dataTs = [xtr3D,ytr],[xts3D,yts]
## ---- export datasets for Jupyter notebook -----
np.save('xTrSF_shiftLSTM-PrStep'+str(Tsteps)+'_'+str(preStep),xtr3D)
np.save('yTrSF_shiftLSTM-PrStep'+str(Tsteps)+'_'+str(preStep),ytr)
np.save('xTsSF_shiftLSTM-PrStep'+str(Tsteps)+'_'+str(preStep),xts3D)
np.save('yTsSF_shiftLSTM-PrStep'+str(Tsteps)+'_'+str(preStep),yts)
### ---- Build and train network: LSTM ----
lstm = lstmNET(net_IHHO,batchSize,fback,Tsteps,Winit,Binit)
lstm.compile(alpha,Tsteps,cost,opti,acti,fback,outMode,preStep)
model= lstm.train(dataTr,dataTs,epoch,batchSize)
# --- export/save model ----
# np.save("ESFLSTM_model"+str(today)[:10],model) # save model # --- save model
# ---- Model performance -----
thresh = 0.5   # threshold 
ymodel_Ts = model['yTs_pre']
yTs_pre  = ymodel_Ts.copy()
yTs_act  = model['yTs_act'].copy()
# --- Classification performance ---
yTs_pre[yTs_pre>=thresh] = 1
yTs_pre[yTs_pre <thresh] = 0
acc = sum(yTs_pre == yTs_act)/len(yTs_act)
print("Accuracy: "+str("{:.4f}".format(acc[0]*100))+"%")  # show accuracy 
# ---- PLOT PREDICTED AND ACTUAL OUTPUTS ----
plots.plotMvsY(yTs_act,ymodel_Ts)
# ========= load dataset for a prediction ===========
dataset_xy = np.load('dataset_202209.npy',allow_pickle=True).tolist()  # load data
data_x = dataset_xy['xdata'][:,features1].copy()
data_y = dataset_xy['ydata'].copy()
data_x = process.scale(data_x,'minmax')   # normalyzing data
data_y = process.scale(data_y,'minmax')   # normalyzing data
data3D_x = process.reshape2Dto3D(data_x,Tsteps) # reshape data
data_y = process.shiftTarget(data_y,preStep)
## --- export datasets for Jupyter notebook -----
np.save("data09_x"+str(Tsteps),data3D_x)  # save data for keras's test
np.save("data09_y"+str(Tsteps),data_y)
# ======= Make a prediction using model ======
ymodel_2 = model['predict'](data3D_x)        
ymodel_2 = ymodel_2[:-preStep].copy()
y_pred = ymodel_2.copy()
y_pred[y_pred>=thresh] = 1      # thresholding output into class 0 or class 1
y_pred[y_pred<thresh]  = 0
y_tar  = data_y[:-preStep].copy()    # get y-target based on prediction steps ahead
y_pred = y_pred.copy()          # as well as preidicted target
# --- compute total accuracy
acc_pre = sum(y_pred==y_tar)/len(y_tar)    
print(">> Prediction accuracy: "+str("{:.4f}".format(acc_pre[0]*100))+"%") # show accuracy
# ---- PLOT PREDICTED AND ACTUAL OUTPUTS ----
plots.plotMvsY(y_tar,ymodel_2)
# ---- bar plot final results ----
yearset = ['20220906','20220907','20220908','20220909','20220910','20220914','20220915','20220916','20220917','20220920','20220921',
           '20220922','20220925','20220926','20220928','20220929','20220930']
Nday = len(yearset[1:]) 
y_true = y_tar.reshape([Nday,52]).T
y_hat  = y_pred.reshape([Nday,52]).T
# --- compute ESF percentage ---
ytr_per  = (np.sum(y_true,axis=0)/52)*100
yhat_per = (np.sum(y_hat,axis=0)/52)*100
# ---- BAR PLOT -----
plots.barMvsY(ytr_per,yhat_per,yearset[1:])
plots.plotError(ytr_per,yhat_per,yearset[1:])
# ========= load dataset for a prediction ===========
dataset_xy2 = np.load('dataset_202003.npy',allow_pickle=True).tolist()  # load data
data_x2 = dataset_xy2['xdata'][:,features1].copy()
data_y2 = dataset_xy2['ydata'].copy()
data_x2 = process.scale(data_x2,'minmax')   # normalyzing data
data_y2 = process.scale(data_y2,'minmax')   # normalyzing data
data3D_x2 = process.reshape2Dto3D(data_x2,Tsteps) # reshape data
data_y2 = process.shiftTarget(data_y2,preStep)
## --- export datasets for Jupyter notebook -----
np.save("data03_x"+str(Tsteps),data3D_x2) # save data for keras's test
np.save("data03_y"+str(Tsteps),data_y2)
# ======= Make a prediction using model ======
ymodel_22 = model['predict'](data3D_x2)        
ymodel_22 = ymodel_22[:-preStep].copy()
y_pred2 = ymodel_22.copy()
y_pred2[y_pred2>=thresh] = 1      # thresholding output into class 0 or class 1
y_pred2[y_pred2<thresh]  = 0
y_tar2  = data_y2[:-preStep].copy()    # get y-target based on prediction steps ahead
y_pred2 = y_pred2.copy() # as well as preidicted target
# compute total accuracy
acc_pre2 = sum(y_pred2==y_tar2)/len(y_tar2)    
print(">> Prediction accuracy: "+str("{:.4f}".format(acc_pre2[0]*100))+"%") # show accuracy
# ---- PLOT PREDICTED AND ACTUAL OUTPUTS ----
plots.plotMvsY(y_tar2,ymodel_22)
# ---- bar plot final results ----
yearset2 = ['20200301','20200302','20200303','20200304','20200305','20200306','20200307','20200308','20200309','20200311','20200312',
           '20200313','20200314','20200315','20200316','20200317','20200318','20200319','20200320','20200321','20200322','20200324',
           '20200325','20200326']
Nday2 = len(yearset2[1:])  
y_true2 = y_tar2.reshape([Nday2,52]).T
y_hat2  = y_pred2.reshape([Nday2,52]).T
# --- compute ESF percentage ---
ytr_per2  = (np.sum(y_true2,axis=0)/52)*100
yhat_per2 = (np.sum(y_hat2,axis=0)/52)*100
# ---- BAR PLOT -----
plots.barMvsY(ytr_per2,yhat_per2,yearset2[1:])
plots.plotError(ytr_per2,yhat_per2,yearset2[1:])

print('>> Training model has been completed ....')





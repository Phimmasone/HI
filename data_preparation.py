import numpy as np
import matplotlib.pyplot as plt
from generateINPUTS import generates as gens
from readESFDATA_MAT import getESFdata
import os
def prepareDATA(*arg):
    '''This function is to match data between inputs and target based on year, month, day. Also, it processes input parameters.
    Finally, it exports or saves shaped dataset to outside. 
    '''
    # --- Path ---
    data_path = os.getcwd()[:-9]+"dataset\\"
    # --- time ---
    time_utc = np.arange(0,24,1/4)
    tpss     = time_utc >= 11
    time_17  = np.where(time_utc==17)
    time_11  = np.where(time_utc==11)
    # --- h'F and foF2 data ---
    hF_path = data_path+"\\Iono_indices\\hFdata.npy"
    hF_var  = np.load(hF_path,allow_pickle=True).tolist()
    hF_ymd  = hF_var['ymd_ind']
    hF_var  = hF_var['hF_data'].T
    f2_path = data_path+"\\Iono_indices\\foF2data.npy"
    f2_var  = np.load(f2_path,allow_pickle=True).tolist()
    f2_ymd  = f2_var['foF2_ymd']
    f2_var  = f2_var['foF2_data'].T
    # ---- ESF data ----
    ESFdata = getESFdata()  # get ESF dataset
    YMD_ESF = ESFdata['YMD']
    ESFdata = ESFdata['ESF_data15']
    ESFdata[np.isnan(ESFdata)] = 0
    # ---- Solar and Magnetic indices ----
    SM_path = data_path+"Space_indices\\Indices_Kp8ap8ApSNF107F107adj.npy"
    SM_vars = np.load(SM_path,allow_pickle=True).tolist()
    SM_ymd  = [val for val in SM_vars.keys()]
    # --- Matching YMD of ESFdata and indices ---
    YMDSF_ind,YMDSM_ind,YMDhF_ind,YMDf2_ind = [],[],[],[]
    for i,YMDval in enumerate(YMD_ESF):
        if YMDval in hF_ymd:
            YMDSF_ind.append(i) # get index for ESF
            YMDhF_ind.append(hF_ymd.index(YMDval)) # get index of matched y_m_d for h'F
            YMDSM_ind.append(SM_ymd.index(YMDval)) # get index of matched y_m_d for Solar&Magnetic
        if YMDval in f2_ymd:
            YMDf2_ind.append(f2_ymd.index(YMDval)) # get index of matched y_m_d for foF2
    # --- Get vertical drift velocity of F-layer ---
    Vz_pss,hF_pss = [],[]
    Vz = []
    for i,ind in enumerate(YMDhF_ind):
        Vz.append(gens.driftVZ(hF_var[:,ind],1))
        Vz_pss.append(Vz[i].copy())
        hF_pss.append(hF_var[:,ind].copy())
        # --- smallize data in given time period ----
        Vz_pss[i][:time_11[0][0]] *= 9e-1
        Vz_pss[i][time_17[0][0]:] *= 9e-1
    # --- Get power spectrum of the GWs ----
    t_GW = np.zeros_like(time_utc)
    for i,val in enumerate(time_utc):
        if val>=11 and val<=21: t_GW[i] = True
        else: t_GW[i] = False
    t_GW = (t_GW == 0)  # get index
    GW = []
    for i,ind in enumerate(YMDf2_ind):
        print(">> Calculate GWs "+str(i+1)+"/"+str(len(YMDf2_ind)))
        gwval = gens.wavelet(f2_var[:,ind])
        GW.append(gwval.copy())
        GW[i][t_GW] *= 9e-1
    # --- insert data by time index and y-m-d index ---
    hFset = np.zeros([sum(tpss),len(YMDhF_ind)])
    f2set,SFset = np.zeros_like(hFset),np.zeros_like(hFset)
    GWset,Vzset = np.zeros_like(hFset),np.zeros_like(hFset)
    ap8,Ap = np.zeros_like(SFset),np.zeros_like(SFset)
    kp8,Kp = np.zeros_like(SFset),np.zeros_like(SFset)
    SSN,F10 = np.zeros_like(SFset),np.zeros_like(SFset)
    F10adj = np.zeros_like(SFset)
    VzpssSet,hFpssSet = np.zeros_like(hFset),np.zeros_like(hFset)
    medhF,medVz,medf2,medGW = [],[],[],[]
    for i,ind in enumerate(YMDhF_ind):  # hF
        hFset[:,i]    = hF_var[tpss,ind].copy()
        Vzset[:,i]    = Vz[i][tpss].copy() # Vz
        Vzset[-1,i]   = 9e-1
        VzpssSet[:,i] = Vz_pss[i][tpss].copy()
        hFpssSet[:,i] = hF_pss[i][tpss].copy()
    for i,ind in enumerate(YMDf2_ind):  # foF2
        f2set[:,i] = f2_var[tpss,ind].copy()
        GWset[:,i] = GW[i][tpss].copy()  # GW
    for i,ind in enumerate(YMDSF_ind):  # ESF
        SFset[:,i] = ESFdata[tpss,ind].copy()
    for i,ind in enumerate(YMDSM_ind):
        kp8v  = SM_vars[SM_ymd[ind]][0] # Kp8
        ap8v  = SM_vars[SM_ymd[ind]][1] # ap8
        Apv   = SM_vars[SM_ymd[ind]][2] # Ap
        SNv   = SM_vars[SM_ymd[ind]][3] # SSN
        F10v  = SM_vars[SM_ymd[ind]][4] # F10.7
        F10a  = SM_vars[SM_ymd[ind]][5] # F10.7adj
        kp8[:,i] = np.repeat(kp8v,12)[tpss]  # repeat and put the kp8v into the kp8
        Kp[:,i]  = np.repeat(round(np.mean(kp8v)),sum(tpss)) # repeat and average the kp8v into the Kp
        ap8[:,i] = np.repeat(ap8v,12)[tpss]  # repeat and put the ap8v into the ap8
        Ap[:,i]  = np.repeat(Apv,sum(tpss))  # repeat and put the Apv into the Ap
        SSN[:,i] = np.repeat(SNv,sum(tpss))  # repeat and put the SN into the SSN
        F10[:,i] = np.repeat(F10v,sum(tpss)) # repeat and put the F10 into the F10
        F10adj[:,i] = np.repeat(F10a,sum(tpss)) # repeat and put the F10a into the F10adj
    # --- Seasonal variations ----
    YMD_main = [YMD_ESF[ind] for ind in YMDSF_ind]
    DOY_set = []
    for i,val in enumerate(YMD_main):
        _,doy = gens.convYMD2DOY(val[:4],val[4:6],val[6:])
        DOY_set.append(int(doy))
    DnS,DnC = gens.daySC(np.array(DOY_set))
    DnS = np.repeat(DnS,sum(tpss)).reshape(SFset.shape)
    DnC = np.repeat(DnC,sum(tpss)).reshape(SFset.shape)
    # --- Diurnal variations ---
    HrS,HrC = gens.timeSC(time_utc[tpss])
    HrS = np.repeat(HrS,len(DOY_set)).reshape(SFset.shape)
    HrC = np.repeat(HrC,len(DOY_set)).reshape(SFset.shape)
    # --- store dataset -----
    dataset = {}
    xdata = np.zeros([SFset.size,16])
    xdata[:,:] = np.array([HrS.T.flatten(),HrC.T.flatten(),DnS.T.flatten(),DnC.T.flatten(),kp8.T.flatten(),Kp.T.flatten(),
                          ap8.T.flatten(),Ap.T.flatten(),SSN.T.flatten(),F10.T.flatten(),F10adj.T.flatten(),Vzset.T.flatten(),
                          GWset.T.flatten(),hFset.T.flatten(),hFpssSet.T.flatten(),VzpssSet.T.flatten()]).T
    dataset['xdata'] = xdata.copy()
    dataset['ydata'] = SFset.T.flatten()
    dataset['features'] = "0.HrS, 1.HrC, 2.DnS, 3.DnC, 4.kp8, 5.Kp, 6.ap8, 7.Ap, 8.SSN, 9.F10.7, 10.F10.7adj, 11.Vz, 12.GW, 13.h'F, 14.h'Fpss, 15.Vzpss"
    # --- Save dataset ----
    file2save_tr = "dataset_trainV01"
    np.save(data_path+"Training_set\\"+file2save_tr,dataset) # save dataset with combined parameters
    return dataset
# esfdata = prepareDATA()


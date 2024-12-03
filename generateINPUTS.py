import numpy as np
import os
from ftplib import FTP
import ftp_download as ftpd
from datetime import datetime
import data_process as dprocess
'''
This contains functions for input generators:
'''
class generates(object):
    def __init__(*arg):
        generates.function_descriptions = 'Contains many functions to mainly process data and generate input parameters'
    # ==== linearly interpolate NaN in data ====
    def interpNAN(x,*arg):
        '''Linearly interpolation function to convert the NAN values into numbers based on the conditions'''
        # x[x==0] = np.nan
        if sum(np.isnan(x))!=96:
            x_range = np.arange(len(x))
            nan_ind = np.where(np.isnan(x))
            for p in range(len(nan_ind[0])):
                i = nan_ind[0][p]
                if i==0:
                    k=i+1
                    while np.isnan(x[i]):
                        if ~np.isnan(x[k]): x[i]=x[k]
                        else: k+=1
                elif i==np.max(x_range):
                    k=i-1
                    while np.isnan(x[i]):
                        if ~np.isnan(x[k]): x[i]=x[k]
                        else: k-=1
                elif i>0 and i<np.max(x_range):
                    if ~np.isnan(x[i-1]) and ~np.isnan(x[i+1]): x[i] = (x[i-1]+x[i+1])/2
                    elif ~np.isnan(x[i-1]) and np.isnan(x[i+1]): x[i] = x[i-1]
                    elif np.isnan(x[i-1]) and ~np.isnan(x[i+1]): x[i] = x[i+1]
                if np.nan in x:
                    print('>> NaN values are still found in array ...!!!')
                    ValueError('Please check your interpNAN fucntion ......!!! ')
        else:
            x[np.isnan(x)] = 0
        return x
    # ==== Create diurnal variations by sine and cosine components ====
    def timeSC(HrN,*arg):
        '''Sine and Cosine component functions to convert any number into the cyclic components'''
        hrs = np.sin((2*np.pi*HrN)/24)
        hrc = np.cos((2*np.pi*HrN)/24)
        return hrs,hrc
    # ==== Create sesonal variations by sine and cosine components ====
    def daySC(DN,*arg):
        '''Sine and Cosine component functions to convert any number into the cyclic components'''
        dns = np.sin((2*np.pi*DN)/365.25)
        dnc = np.cos((2*np.pi*DN)/365.25)
        return dns,dnc
    # ==== Compute drift velocity by dhF/dt ====
    def driftVZ(hF,w=1,*arg):
        '''Differentiative function to compute veritical drift velocity as dh'F/dt '''
        Vz  = np.zeros_like(hF)
        dt  = w*15*60  # get time in second unit
        hF *= 1000  # get height in metre unit
        hF = generates.interpNAN(hF)
        for i in range(0,len(hF)-1,1):
            dh = (hF[i+1] - hF[i])
            Vz[i] = dh/dt
        return Vz
    # ==== Convert Year Month Day to Day of year ====
    def convYMD2DOY(year,month,day,*arg):
        '''Function to convert YMD (Year Month Day) to DOY (Day of year)'''
        doy = datetime(int(year),int(month),int(day)).timetuple().tm_yday
        return [year,str(doy)]
    # ==== Convert Day of year to Year Month Day ====
    def convDOY2YMD(year,doy,*arg):
        '''Function to convert DOY (Day of year) to YMD (Year Month Day)'''
        doy1 = str(doy)
        doy1.rjust(3+len(doy1),'0')
        d_m_y = datetime.strptime(str(year)+"-"+doy1, "%Y-%j").strftime("%m-%d-%Y")
        return d_m_y
    # ===== Calculate Wavelet analysis ======
    def DFT(x,*arg):  # Discrete Fourier Transform
        '''Discrete Fourier Transform (DFT)'''
        N  = len(x)
        xn = x.copy()
        xk_hat = np.zeros(N,dtype=complex)
        for k in range(0,N):
            x_fft = 0.0
            for n in range(0,N):
                x_fft += xn[n] * np.exp((-2j*np.pi*k*n)/N)
            xk_hat[k] = x_fft/N
        return xk_hat
    # ==== Calculate Wavelet signal ====
    def morlet0(s,wk,*arg):
        '''Morelet wavelet function'''
        w = wk.copy()
        scale = s.copy()
        w0 = 6.0    # non-dimensional frequency
        psi0_hat = np.pi**(-1/4) * generates.H(w) * np.exp(-(scale*w-w0)**2 / 2)
        return psi0_hat
    # ==== Generate Heaviside signal ====
    def H(w,*arg):  # Heaviside step function from Table 1
        '''Heaviside step function returns output as 0 or 1'''
        wk = w.copy()
        wk[wk>0] = 1.
        wk[wk<=0] = 0.
        return wk
    # ==== Calculate Wavelet transform ====-
    def wavelet(x,dt=1/4,*arg):
        '''This is Wavelet analysis function based on paper of ( C. Torrence, 1998)'''
        N  = len(x)  # length of signal
        w0 = 6.0     # fixed frequency
        s0 = 2*dt    # smallest scale
        dj = 1/4     # finer resolution < 0.5 for Morlet
        J  = dj**(-1) * np.log2(N*dt/s0)   # eq[10]
        j1 = np.arange(0,J)
        scale = s0 * 2**(j1*dj)            # eq[9]
        # ==== Inpterpolate NAN of signal ====- 
        x = x/100  # divide x by 100 to get MHz
        xn  = generates.interpNAN(x)
        xn  = x.copy() - np.mean(x)   # remove mean value        
        # ==== Define zero padding: power of 2 nearest to N
        base2   = np.fix(np.log(N) / np.log(2) + 0.4999)
        nzeroes = (2 ** (base2 + 1) - N).astype(np.int64)
        xn = np.concatenate((xn, np.zeros(nzeroes)))   # pad zeros into xn
        # compute DFT of xn using eq[3]
        xk_hat = generates.DFT(xn)        # eq[3] or np.fft.fft(xn) can be used
        # construct wk using eq[5]
        k  = np.arange(0,len(xk_hat),1)
        wk = np.zeros_like(k,dtype=float)
        for i in range(len(k)):
            if k[i] <= len(xk_hat)/2:
                wk[i] = (2*np.pi*k[i])/(len(xk_hat)*dt)
            elif k[i] > len(xk_hat)/2:
                wk[i] = -(2*np.pi*k[i])/(len(xk_hat)*dt)
        # wk[int(len(xk_hat)/2):] = np.sort(wk[int(len(xk_hat)/2):])    
        wave = np.zeros([scale.size,len(xk_hat)],dtype=complex)   # wave array
        for i,s in enumerate(scale):        # loop through all scales
            norm = ((2*np.pi*s)/dt)**(1/2)  
            # norm = np.sqrt(s * wk[1]) * (np.pi ** (-0.25)) * np.sqrt(N)
            psi0_hat = generates.morlet0(s, wk)       # table 1
            psi_hat  = norm * psi0_hat      # eq[6]    
            for n in range(0,len(xk_hat)):
                w_ifft = 0.0
                for k in range(0,len(xk_hat)):
                    w_ifft += xk_hat[k] * (psi_hat[k] * np.exp(1j*wk[k]*n*dt))  #eq[4]
                wave[i,n] = w_ifft    
        Fourier_wavelength = (4 * np.pi * scale) / (w0 + np.sqrt(2 + w0**2))       # Table 1
        Freq   = 1/Fourier_wavelength  # get frequencies
        period = 1/Freq       # t=1/f; where t is periodicity
        # coi    = Fourier_wavelength/np.sqrt(2) 
        wave   = wave[:,:N]   # get rid of padding before returning
        power  = (np.abs(wave))**2     # get power spectrum 
        # ==== Get average of the Wavelet power spectrum by given periodicity ====-
        period_ind = period<1.6   # 1.6 hours mean to 96.0 min
        avg_GW     = np.mean(power[period_ind,:], axis=0) # compute avg. of power   
        return avg_GW
    # === Automatically download Space indices ====
    def loadINDICES(year,dst_dir='',*arg):
        '''This function is to automatically download datatset from WDC if desired data can not be found'''
        if not dst_dir: dst_dir= os.getcwd()+"\\dataset\\Space_indices\\"
        file_name = "Kp_ap_Ap_SN_F107_"+year+".txt" # Name format: Kp_ap_Ap_SN_F107_YEAR.txt
        ftp_host = "ftp.gfz-potsdam.de"
        data_dir = "/pub/home/obs/Kp_ap_Ap_SN_F107/"
        ftp = FTP(ftp_host)     # Connect to the FTP server
        ftp.login()  # Anonymous login
        ftp.cwd(data_dir)       # Change to the target directory
        # ftp.retrlines("LIST") # list files in directory
        # files = ftp.nlst()    # List files in the directory (to check if file exists)
        file = dst_dir+file_name
        if not os.path.exists(file):
            if file_name in ftp.nlst(): ftpd.file(ftp,file_name,dst_dir)
            else: print(">> File name : "+file_name+" does not exist...!!!")
        elif os.path.exists(file): print('>> File name ['+ file_name +'] already exists in '+dst_dir)
        ftp.quit()
    # === Read indices from text file ===
    def readINDICEtxt(file_path,yrnew,*arg):
        '''This function is to read solar and magnetic indices from text format file.'''
        dataset_out = {}
        if not file_path: file_path = os.getcwd()[:-9]+"dataset\\Space_indices"
        yrs_set = [str(val) for val in np.arange(2008,2024)]    
        if yrnew.isalnum() and len(yrnew)==4: yrs_set.append(yrnew)
        else: print('>> Year is incorrect ...'+yrnew)
        for yrs in yrs_set:
            # print(yrs)
            if type(yrs) is not str: yrs = str(yrs)
            file_name = "Kp_ap_Ap_SN_F107_"+yrs+'.txt'
            fullname  = file_path+'\\'+file_name
            if os.path.exists(fullname):
                f_open = open(fullname,"r")
                fread = f_open.readlines()
                dn = len(fread) - 40
                for i in range(0,dn,1):
                    f_txt = fread[i+40]
                    y,m,d = f_txt[:4],f_txt[5:7],f_txt[8:10]
                    Kp8 = [float(val) for val in f_txt[34:88].split()]
                    ap8 = [int(val) for val in f_txt[89:128].split()]
                    Ap  = [int(val) for val in f_txt[131:134].split()]
                    SN  = [int(val) for val in f_txt[136:138].split()]
                    F10 = [float(val) for val in f_txt[139:147].split()]
                    F10adj = [float(val) for val in f_txt[149:156].split()]
                    dataset_out[y+m+d] = [Kp8,ap8,Ap,SN,F10,F10adj]
            elif not os.path.exists(fullname):
                print('>> File does not exist in '+fullname)
        dataset_out['description'] = ["Kp8","ap8","Ap","SN","F10.7","F10.7adj"]
        np.save(file_path+"\\Indices_Kp8ap8ApSNF107F107adj",dataset_out)
        return dataset_out
    # === Generate Xdata for prediction ===
    def genINPUT(yearset,*arg):
        '''Load and generate inputs for model by inputing the CPNFYYYYMMDD, i.e. CPNFC20220301'''
        path = os.getcwd()[:-9]
        time = np.arange(0,24,1/4)
        tpss,t11,t17 = time>=11,np.where(time==11),np.where(time==17)
        f_path = path+'\\dataset\\Predictions\\Xdatasets\\'
        fymd   = np.loadtxt(f_path+'CPNFC'+yearset[0][:6]+'_ymd.csv',delimiter=",")
        f_ymd = []
        for _,val in enumerate(fymd):
            f_ymd.append(''.join(str(int(v)) for v in val))
        hfdata = np.loadtxt(f_path+'CPNFC'+yearset[0][:6]+'_hF.csv',delimiter=",")
        f2data = np.loadtxt(f_path+'CPNFC'+yearset[0][:6]+'_foF2.csv',delimiter=",")
        sfdata = np.loadtxt(f_path+'CPNFC'+yearset[0][:6]+'_ESF.csv',delimiter=",")
        # --- Get Solar and Magnetic indices ---
        xdata   = np.zeros([sum(tpss)*len(yearset),len(yearset)])
        dataset = {}
        hf,gw,vz,hs,hc,ds,dc,f10,ap,sf,vz0,hf0,kp,ssn = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        for _,ymd in enumerate(yearset):
            SM_path = '\\dataset\\Space_indices\\Indices_Kp8ap8ApSNF107F107adj.npy'  # this file is provided by 'readINDICEtxt'
            SM_indices = np.load(path+SM_path,allow_pickle=True).tolist()
            if ymd in SM_indices.keys(): # if year month day exists
                F10 = np.array(SM_indices[ymd][4])
                SN = np.array(SM_indices[ymd][3])
                Ap8 = np.array(SM_indices[ymd][1])
                Kp8 = np.array(SM_indices[ymd][0])
            elif ymd not in SM_indices.keys(): # if year month day do not exist...Then dowload file and read
                generates.loadINDICES(ymd[:4])   # automatically download the data via FTP
                generates.readINDICEtxt('',[ymd[:4]]) # read text file and export to directory
                print(">> Please re-run this fucntion again .....")
            # --- get time and day ---
            hrs,hrc = generates.timeSC(time[tpss])
            _,doy   = generates.convYMD2DOY(ymd[:4],ymd[4:6],ymd[6:])
            dns,dnc = generates.daySC(int(doy))
            # --- get ionospheric indices ----
            f_ind = f_ymd.index(ymd)
            hfval = generates.interpNAN(hfdata[f_ind,1:].copy()/1000)
            vzval = generates.driftVZ(hfval,1)
            vz00  = vzval[tpss].copy()
            hf00  = hfval[tpss].copy()
            vzval[:t11[0][0]] *= 9e-1
            vzval[t17[0][0]:] *= 9e-1
            f2val = generates.interpNAN(f2data[f_ind,1:].copy())
            gwval = generates.wavelet(f2val,1/4)
            gwval[:t11[0][0]] *= 9e-1
            gwval[t17[0][0]:] *= 9e-1
            sfval = sfdata[f_ind,1:].copy()
            sfval[sfval==1] = 0
            sfval[sfval>0]  = 1
            sfval[np.isnan(sfval)] = 0
            # --- repeat data ----
            F10,Ap8 = np.repeat(F10,sum(tpss)),np.repeat(Ap8,12)
            SN,Kp8  = np.repeat(SN,sum(tpss)),np.repeat(Kp8,12)
            dns,dnc = np.repeat(dns,sum(tpss)),np.repeat(dnc,sum(tpss))
            hs.append(hrs)
            hc.append(hrc)
            ds.append(dns)
            dc.append(dnc)
            f10.append(F10)
            ssn.append(SN)
            ap.append(Ap8[tpss])
            kp.append(Kp8[tpss])
            gw.append(gwval[tpss])
            vz.append(vzval[tpss])
            sf.append(sfval[tpss])
            hf.append(hfval[tpss])
            hf0.append(hf00)
            vz0.append(vz00)
        # --- put data into format ---
        # "0.HrS, 1.HrC, 2.DnS, 3.DnC, 4.kp8, 5.ap8, 6.SSN, 7.F10.7adj, 9.hF, 10.Vz, 11.hftpss, 12.Vztpss, 13.GWtpss
        in1,in2   = np.array(hs).flatten(),np.array(hc).flatten()
        in3,in4   = np.array(ds).flatten(),np.array(dc).flatten()
        in5,in6   = np.array(kp).flatten(),np.array(ap).flatten()
        in7,in8   = np.array(ssn).flatten(),np.array(f10).flatten()
        in9,in10  = np.array(hf0).flatten(),np.array(vz0).flatten()
        in11,in12 = np.array(hf).flatten(),np.array(vz).flatten()
        in13 = np.array(gw).flatten()
        xdata = np.array([in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11,in12,in13]).T
        ydata = np.array(sf).flatten()
        if ydata.ndim==1:
            ydata = np.expand_dims(ydata,1)
        dataset['xdata'] = xdata.copy()
        dataset['ydata'] = ydata.copy()
        np.save(path+"dataset_"+yearset[0][:6],dataset)
        return dataset
# a = generates.readINDICEtxt('','')
yearset = ['20220906','20220907','20220908','20220909','20220910','20220914','20220915','20220916','20220917','20220920','20220921',
           '20220922','20220925','20220926','20220928','20220929','20220930']
# yearset = ['20200301','20200302','20200303','20200304','20200305','20200306','20200307','20200308','20200309','20200311','20200312',
#            '20200313','20200314','20200315','20200316','20200317','20200318','20200319','20200320','20200321','20200322','20200324',
#            '20200325','20200326']
dataset = generates.genINPUT(yearset)
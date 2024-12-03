import numpy as np
import scipy.io as sio
import os
from datetime import datetime
today = datetime.today()
'''
This function loads dataset.mat of ESF and returns dataset and thier ymd (Year Month Day).
    > newyearAdd: if you added new ESF data, you need to identify year out of 'yearset'
    > src_dir: path to directory which contains ESF_file.mat
    > dst_dir: path to destination directory for saving readed ESF data
'''
def getESFdata(newyearAdd=[],src_dir=[],dst_dir=[],file2save=[],*arg,**kwarg):
    yearset = ['2008', '2009', '2010', '2011', '2012', '2013',
               '2014', '2015', '2016', '2018', '2019']  # these are used in loopings
    if newyearAdd not in yearset:
        if not newyearAdd: pass
        else: yearset.append(newyearAdd)
    if not src_dir: src_dir  = 'D:\\KMITL\\DR_research_experments\\Experiments\\Data_preparation\\'  # get the current run program
    if not dst_dir: dest_dir = os.getcwd()[:-9]+'dataset\\ESF_target\\'
    if not file2save: file2save = 'ESF_DATASET_TR'
    ESF_DIR   = 'ESF_data'  # path to ESF folder
    ymd_index = []  # store ymd index in this list
    ESF_data  = []
    for item in yearset:  # looping through years
        ESF_filename = 'CPN' + item + 'ESF_data.mat'
        ESF_fullpath = src_dir + ESF_DIR + '\\' + ESF_filename
        ESF_loaded = sio.loadmat(ESF_fullpath)
        ESF = ESF_loaded['ESFdata']
        for mth in range(ESF[:, 0].size):  # looping through all months
            ESF_m, ymd = ESF[mth,0], ESF[mth, 1]
            if ESF_m.size != 0: ESF_data.append(ESF_m)
            if ymd.size != 0: ymd_index.append([str(val) for val in ymd])  # put ymd index into list        
    ESF_val = ESF_data[0]  # murge all ESF days together in one
    for i in range(len(ESF_data)-1):
        ESF_val = np.concatenate((ESF_val, ESF_data[i+1]), axis=1)  # connecting
    freq_SF, range_SF, strong_SF, mixed_SF = [1, 2, 3, 4]  # each type of ESFs
    time15 = np.arange(0, 24, 0.25, dtype = float)  # time referrence in every 15 minutes
    time30 = np.arange(0, 24, 0.5, dtype = float)   # time referrence in every 30 minutes
    time30_ind = np.zeros(time30.size, dtype = int)  # collect time index here
    for i in range(time30.size):  # loop through each time step
         time30_ind[i] = int(np.where(time30[i] == time15)[0][0])  # get time index of 30min in 15min format
    for d in range(ESF_val.shape[1]):
        find_freq   = np.where(ESF_val[:, d] == freq_SF)
        find_range  = np.where(ESF_val[:, d] == range_SF)
        find_strong = np.where(ESF_val[:, d] == strong_SF)
        find_mixed  = np.where(ESF_val[:, d] == mixed_SF)
        ESF_val[find_freq, d]   = np.nan
        ESF_val[find_range, d]  = 1
        ESF_val[find_strong, d] = 1
        ESF_val[find_mixed, d]  = 1
    ESF_val30 = ESF_val[time30_ind,:]   # choose ESF depending on 30 min sample time 
    ESF_ymd   = []  # list array to store index
    month_letter = ['FEB', 'MAR', 'APR', 'AUG', 'SEP', 'OCT']  # these are used to convert Feb to 02
    month_num = ['02', '03', '04', '08', '09', '10']  # these are used with mentioned above
    for i in range(len(ymd_index)):  # loop through all obtained index from ESF data
        ymd_i = ymd_index[i]
        for val in ymd_i:  # get each item of entire index
            if len(val) > 8:  # check length if length != 8 
                if val.isnumeric() == False:
                    if 'CP' in val:  # check if letter 'CPN' exists in ymd
                        ymd_n1 = val.replace('CP', '')  # replace 'CPN' with '' (empty)
                    if 'N' in ymd_n1:  # check if letter 'FC' exists in ymd 
                        ymd_n1 = ymd_n1.replace('N', '')  # replace 'FC' with ''
                    if 'FC' in ymd_n1:
                        ymd_n1 = ymd_n1.replace('FC', '')
                    # for j, m in enumerate(month_letter):  # loop through month referrence for conversion
                    if ymd_n1.isnumeric() ==  True:
                        ESF_ymd.append(ymd_n1[:8])
                    elif ymd_n1.isnumeric() == False:  # check if there is month letter then convert it
                        for j, m in enumerate(month_letter):
                            if m in ymd_n1:
                                ymd_n2 = ymd_n1.replace(month_letter[j], month_num[j])  # replace letter with number instead
                                ESF_ymd.append(ymd_n2[:8])  # put new ymd index to new one
                elif val.isnumeric() == True:
                    ESF_ymd.append(val[:8])
                    val = []
            elif len(val) == 8:  # normally, ymd = 20080301 (length = 8)
                ESF_ymd.append(val)  # put ymd into new ymd index
                val = []
            ymd_n1, ymd_n2 = [], []
    ESF_ymd = [str(val) for val in ESF_ymd if val != []]
    dataset = {'ESF_data15':ESF_val,'YMD':ESF_ymd,'ESF_data30':ESF_val30,'Time15':time15,'Time30':time30}  # store dataset and ymd index into dict
    np.save(dest_dir+file2save+str(today)[:10],dataset)
    return dataset
# ESFdata = readESFDATA_MATFILE()

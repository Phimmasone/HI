import matplotlib.pyplot as plt
import numpy as np

class plots(object):
    def plotMvsY(y_target,y_model,*arg):
        # --- convert predicted to class 0 or 1 -----
        y_pred = y_model.copy()
        y_pred[y_pred>=0.5] = 1
        y_pred[y_pred<0.5]  = 0
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(y_target,'-'),plt.title("Observed ESF")
        plt.plot(y_model,'-')
        plt.subplot(3,1,2)
        plt.plot(y_model,'-'),plt.grid(True),plt.legend(['Predicted'])
        plt.subplot(3,1,3)
        plt.plot(y_pred),plt.grid(True),plt.legend(['Predicted with threshold'])
        plt.show()

    def barMvsY(ytr_per,yhat_per,yearset,*arg):
        x_label = yearset  # select YMD based on shifted target
        x_axis = np.arange(0,len(x_label))
        width,multiply = 0.25, 0
        sf_plot = {'Observed ESF': ytr_per.round(1),
                'LSTM model': yhat_per.round(1)}
        # --- BAR PLOT ---
        fig, ax = plt.subplots(layout='constrained')
        for attr,val in sf_plot.items():
            offset = width * multiply
            rects  = ax.bar(x_axis + offset, val, width, label=attr)
            ax.bar_label(rects,padding=3)
            multiply+=1
        ax.set_ylabel('Percentage')
        ax.set_title('Daily prediction ahead')
        ax.set_xticks(x_axis+width,x_label,rotation=75)
        ax.legend(loc='upper left',ncols=2)
        ax.set_ylim([0,100])
        plt.show()

    def plotInputs(data_x,*arg):
        _,s2= data_x.shape
        plt.figure(0)  # training set
        for i in range(s2):
            plt.subplot(4,3,i+1)
            plt.plot(data_x[:,i]),plt.grid(True)
            plt.legend(['Input '+str(i+1)])
        plt.show()

    def plotError(yPr_per,yTr_per,x_label,*arg):
        e = abs(yPr_per-yTr_per)
        Nday = len(e)
        plt.figure()
        plt.bar(np.arange(0,Nday),e)
        plt.ylabel('Percentage'),plt.legend(['error'])
        plt.xticks(np.arange(0,Nday),x_label,rotation=75)
        plt.ylim(0,100),plt.grid(True)
        plt.show()

    def confusion(yPre,yTru,*arg):
        from collections import Counter
        '''Confusion matrix is designed to support classes.'''
        yPre,yTru = np.array(yPre),np.array(yTru)
        cl = Counter(yTru).keys()
        cl = [v for v in cl]
        cl.sort()
        Pre,Act,TP,TN,FP,FN = {},{},{},{},{},{}
        F1 = {}
        for i,c in enumerate(cl):
            Pre['class'+str(c)] = np.where(yPre==c)
            Act['class'+str(c)] = np.where(yTru==c)
        
        acc = sum(yPre==yTru)/len(yPre)
        plt.figure()
        

        plt.show()

# v1 = [0,1,2,3,0,0,1,1,2,3,5,0,5,5,6]
# v2 = [1,0,2,3,1,0,1,5,2,3,5,0,5,5,6]
# con = plots.confusion(v1,v2)
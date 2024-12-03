import numpy as np
import matplotlib.pyplot as plt

# ---- import model ----
class predicts(object):
    def dailyProb(yPre,ymd,*arg):
        y_per = (np.sum(yPre)/len(yPre))*100
        if y_per<=9:
            print('>>> On '+ymd+': ')
            print('   <ESF_c1> Predicted daily ESF: '+str("{:.1f}".format(y_per)))
        elif y_per>9 or y_per<=18:
            print('>>> On '+ymd+': ')
            print('   <ESF_c2> Predicted daily ESF: '+str("{:.1f}".format(y_per)))
        elif y_per>18 or y_per<=26:
            print('>>> On '+ymd+': ')
            print('   <ESF_c3> Predicted daily ESF: '+str("{:.1f}".format(y_per)))
        elif y_per>26 or y_per<=35:
            print('>>> On '+ymd+': ')
            print('   <ESF_c4> Predicted daily ESF: '+str("{:.1f}".format(y_per)))
        elif y_per>35 or y_per<=80:
            print('>>> On '+ymd+': ')
            print('   <ESF_c5> Predicted daily ESF: '+str("{:.1f}".format(y_per)))
        elif y_per>80:
            print('>>> On '+ymd+': ')
            print('   <ESF_c6> Predicted daily ESF: '+str("{:.1f}".format(y_per)))
        else: ValueError('<!!> Given values are not correct ...')
        
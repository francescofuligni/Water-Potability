'''
STUDIO STATISTICO DEI RISULTATI
Script per l'analisi dei dati ottenuti dall'esecuzione del progetto sul dataset
Water Potability con seed differenti
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ingorare FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Campioni di metriche
srs = pd.read_csv('metriche-SVM.csv')
MEs = srs['ME']
MRs = srs['MR']
ACCs = srs['ACC']
# Non considero la metrica Mper

# Massimi
print('\nMax ME: ', np.max(MEs))
print('Max MR: ', np.max(MRs))
print('Max ACC: ', np.max(ACCs))

# Minimi
print('\nMax ME: ', np.min(MEs))
print('Max MR: ', np.min(MRs))
print('Max ACC: ', np.min(ACCs))

# Ordinamento
#MEs = np.sort(MEs)
#MRs = np.sort(MRs)
#ACCs = np.sort(ACCs)





# STATISTICA DESCRITTIVA ******************************************************

# MISURE DEL CENTRO
# Media semplice
mean = [np.mean(MEs), np.mean(MRs), np.mean(ACCs)]
print('\nMEAN')
print(f'\tME: {mean[0]}')
print(f'\tMR: {mean[1]}')
print(f'\tACC: {mean[2]}')

# Mediana semplice
median = [np.median(MEs), np.median(MRs), np.median(ACCs)]
print('\nMEDIAN')
print(f'\tME: {median[0]}')
print(f'\tMR: {median[1]}')
print(f'\tACC: {median[2]}')


# MISURE DELLA DIFFUSIONE DEI DATI
# Varianza
S_2 = [np.var(MEs), np.var(MRs), np.var(ACCs)]
print('\nVARIANCE')
print(f'\tME: {S_2[0]}')
print(f'\tMR: {S_2[1]}')
print(f'\tACC: {S_2[2]}')

# Deviazione standard
S = [np.std(MEs), np.std(MRs), np.std(ACCs)]
print('\nSTANDARD DEVIATION')
print(f'\tME: {S[0]}')
print(f'\tMR: {S[1]}')
print(f'\tACC: {S[2]}')

# IQR
IQR = [np.percentile(MEs,75)-np.percentile(MEs,25), 
        np.percentile(MRs,75)-np.percentile(MRs,25), 
        np.percentile(ACCs,75)-np.percentile(ACCs,25)]
print('\nIQR')
print(f'\tME: {IQR[0]}')
print(f'\tMR: {IQR[1]}')
print(f'\tACC: {IQR[2]}')

# Deviazione assoluta dalla media (MAD)
from statsmodels.robust import mad
MAD = [mad(MEs), mad(MRs), mad(ACCs)]
print('\nMEAN ABSOLUTE DEVIATION')
print(f'\tME: {MAD[0]}')
print(f'\tMR: {MAD[1]}')
print(f'\tACC: {MAD[2]}')


# MISURE DELLA FORMA
from scipy import stats
# Simmetria
g1 = [stats.skew(MEs), stats.skew(MRs), stats.skew(ACCs)]
print('\nSKEWNESS')
print(f'\tME: {g1[0]}')
print(f'\tMR: {g1[1]}')
print(f'\tACC: {g1[2]}')

# Curtosi
g2 = [stats.kurtosis(MEs), stats.kurtosis(MRs), stats.kurtosis(ACCs)]
print('\nKURTOSIS')
print(f'\tME: {g2[0]}')
print(f'\tMR: {g2[1]}')
print(f'\tACC: {g2[2]}')





# GRAFICI ACCURACY ************************************************************

plt.subplot(1,2,1)
plt.scatter(np.arange(len(ACCs)), ACCs, color='red')
plt.title('Accuracy Scatterplot')
plt.xlabel('')
plt.ylabel('Values')

plt.subplot(1,2,2)
sns.boxplot(ACCs, sym='*', color='red')
plt.title('Accuracy Boxplot')
plt.xlabel('')
plt.ylabel('')
plt.show()

plt.subplot(1,2,1)
plt.title('Accuracy KDE')
sns.kdeplot(ACCs, color='red')
plt.xlabel('Acc values')
plt.ylabel('Density')

plt.subplot(1,2,2)
sns.histplot(ACCs, kde=False, linewidth=0.1, bins=len(ACCs), legend=False, color='red')
#sns.rugplot(ACCs, height=0.1, color='red')
plt.title('Accuracy KDE and Histogram')
plt.xlabel('Acc values')
plt.ylabel('')
plt.show()

sns.violinplot(ACCs, color='red')
plt.title('Accuracy Violinplot')
plt.xlabel('Acc values')
plt.show()





# STATISTICA INFERENZIALE *****************************************************
print('\n******************************************************************\n')

# NORMALITÀ DEI CAMPIONI
from scipy.stats import shapiro
stat_ME, p_ME = shapiro(MEs)
stat_MR, p_MR = shapiro(MRs)
stat_ACC, p_ACC = shapiro(ACCs)

print('\nShapiro-Wilk Test')
print(f'\tp_value ME: {p_ME}')
print(f'\tp_value MR: {p_MR}')
print(f'\tp_value ACC: {p_ACC}')

# p > 0.05 -> le metriche hanno distribuzione normale


# INTERVALLI DI CONFIDENZA
# Media campionaria (stmatore non distorto della media della popolazione)
alpha = 0.05
n=len(ACCs)

print('\nConfidence intervals:')
for i in range(0,3):
    
    # Con distribuzione t di Student (n<40)
    Int_conf = stats.t.interval(confidence=1-alpha, df=n-1, loc=mean[i], scale=(S[i]/np.sqrt(n)))

    # Con distribuzione normale (n>40)
    #Int_conf = stats.norm.interval(confidence_level=1-alpha, degree_freedom=1, loc=mean[i], scale=(S[i]/np.sqrt(n)))
    
    if(i==0):
        print(f'\tME: {Int_conf}')
    elif(i==1):
        print(f'\tMR: {Int_conf}')
    else:
        print(f'\tACC: {Int_conf}')

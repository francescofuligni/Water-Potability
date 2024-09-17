'''
Analisi del dataset "water_potability". Il dataset contiene 9 colonne di valori
numerici contenenti informazioni su vari fattori che influenzano la qualità 
dell'acqua e una di interi che classifica il campione come potabile (1) o non 
potabile (0). Le 9 colonne numeriche sono: pH, durezza, solidi, clorammine, 
solfati, conduttività, carbonio organico, trialometani, torbidità. La
potabilità rappresenta la variabile target (categorica).
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ingorare FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# PRE-PROCESSING **************************************************************
# Load data
data=pd.read_csv('water_potability.csv')
#print(data.info())
#print(data.isnull().sum())

# Rimozione valori NaN
data=data.dropna()
#print(data.info())
#print(data.isnull().sum())

# Dati numerici (escludo la variabile di classificazione 'Potability')
numeric_data = data[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']]
data['Potability'] = data['Potability'].astype('category')





# EDA *************************************************************************
numeric_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# Visualizzazione dei dati numerici con boxplot e scatterplot
for c in numeric_columns:
    plt.subplot(1,2,1)
    plt.title(c)
    plt.boxplot(data[c], ".")
    plt.subplot(1,2,2)
    plt.title(c)
    plt.plot(data[c], ".")
    plt.show()
# Non vi sono outliers rilevanti da rimuovere

# Matrice di Correlazione di Pearson
C = numeric_data.corr()
plt.matshow(numeric_data.corr(), vmin=-1, vmax=1)
plt.xticks(np.arange(0, numeric_data.shape[1]), numeric_data.columns, rotation=45)
plt.yticks(np.arange(0, numeric_data.shape[1]), numeric_data.columns)
plt.title('Correlation Matrix')
plt.colorbar()
plt.show()

# Lieve correlazione positiva tra Hardness e ph
# Lieve correlazione negativa tra Sulfate e Solids
# In generale, le variabili sono poco correlate





# SPLITTING *******************************************************************
from sklearn import model_selection

# Seed fissato
np.random.seed(seed=8)

# Divisione del dataset in training set, test set e validation set
# train 70%, test 15%, validation 15%
data_train, data_test = model_selection.train_test_split(data, train_size=1_711)
data_train, data_val = model_selection.train_test_split(data, train_size=1_411)





# REGRESSIONE LINEARE 1 *******************************************************
# Regressione lineare tra Hardness e ph (lievemente correlate positivamente)

from sklearn import linear_model

# Training
x1 = data_train['Hardness'].values.reshape(-1,1)
y1 = data_train['ph'].values.reshape(-1,1)
lrmodel_1 = linear_model.LinearRegression().fit(x1,y1)

# Coefficienti stimati
print('\nB_0 = %.5f' % lrmodel_1.intercept_[0])
print('B_1 = %.5f' % lrmodel_1.coef_[0][0])

# Predizioni sul test set
x1_test = data_test['Hardness'].values.reshape(-1,1)
y1_test = data_test['ph'].values.reshape(-1,1)
y1_pred = lrmodel_1.predict(x1_test)

# Plotting
plt.scatter(x1_test, y1_test)
plt.plot(x1_test, y1_pred, color='red')
plt.xlabel('Hardness')
plt.ylabel('ph')
plt.title('Linear Regression 1')
plt.show()

# Metriche: r^2 e MSE
from sklearn import metrics
print('Mean Squared Error: \t\t\t\t MSE = %.5f' % metrics.mean_squared_error(y1_test, y1_pred))
print('Coefficient of determination: \t r^2 = %.5f' % metrics.r2_score(y1_test, y1_pred))
# r^2 vicino a 0 -> modello poco rappresentativo

# Analisi dei residui con QQ Plot
import statsmodels.api as sm
res1 = y1_test - y1_pred
sm.qqplot(res1[:,0], line='45')
plt.title('QQ Plot Residuals 1')
plt.show()





# REGRESSIONE LINEARE 2 *******************************************************
# Regressione lineare tra Sulfate e Solids (lievemente correlate negativamente)

x2 = data_train['Sulfate'].values.reshape(-1,1)
y2 = data_train['Solids'].values.reshape(-1,1)
lrmodel_2 = linear_model.LinearRegression().fit(x2,y2)

# Coefficienti stimati
print('\nB_0 = %.5f' % lrmodel_2.intercept_[0])
print('B_1 = %.5f' % lrmodel_2.coef_[0][0])

# Predizioni sul test set
x2_test = data_test['Sulfate'].values.reshape(-1,1)
y2_test = data_test['Solids'].values.reshape(-1,1)
y2_pred = lrmodel_2.predict(x2_test)

# Plotting
plt.scatter(x2_test, y2_test, color='green')
plt.plot(x2_test, y2_pred, color='violet')
plt.xlabel('Sulfate')
plt.ylabel('Solids')
plt.title('Linear Regression 2')
plt.show()

# Metriche: r^2 e MSE
print('Mean Squared Error: \t\t\t\t MSE = %.5f' % metrics.mean_squared_error(y2_test, y2_pred))
print('Coefficient of determination: \t r^2 = %.5f' % metrics.r2_score(y2_test, y2_pred))
# r^2 vicino a 0 -> modello poco rappresentativo

# Analisi dei residui con Test di Shapiro-Wilk
from scipy.stats import shapiro
res2 = y2_test - y2_pred
stat, p_value = shapiro(res2)
print('Shapiro-Wilk Test: \t\t\t\t p_value = ' , p_value)
# p < 0.05 -> conferma l'ipotesi nulla H_0 di NON normalità dei residui

# Distribuzione residui
sns.histplot(res2, kde=True, linewidth=0.1, bins=50, legend=False, color='green')
plt.xlabel('Residuals')
plt.ylabel('Density function')
plt.title('Residuals Distribution 2')
plt.show()





# ADDESTRAMENTO DEL MODELLO ***************************************************

# Training set
x_train = data_train[numeric_columns]
y_train = data_train['Potability']

# Test set
x_test = data_test[numeric_columns]
y_test = data_test['Potability']

# Validation set
x_val = data_val[numeric_columns]
y_val = data_val['Potability']
 
# REGRESSIONE LOGISTICA
logreg_model = linear_model.LogisticRegression(solver='lbfgs', max_iter=2_000)
logreg_model.fit(x_train, y_train)

# SVM
from sklearn import svm
#svm_model = svm.SVC(kernel="poly", C=1, degree=4)
#svm_model.fit(x_train, y_train)





# SVM - HYPERPARAMETER TUNING *************************************************
# Utilizzo del validation set

# Scelta del kernel

# Kernel POLY -> scelta del grado
print('\n[POLY] Accuracy with different degree:')
for d in range(1,10):
    svm_model = svm.SVC(kernel='poly', C=1, degree=d)
    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_val)
    ME = np.sum(y_pred != y_val)
    MR = ME/len(y_pred)
    ACC = 1 - MR
    print(f'\tAccuracy [degree={d}]: {ACC}')
    # grado ottimale = 5
    # ACC = 60%
  
# Kernel RBF -> scelta di gamma
print('\n[RBF] Accuracy with different gamma:')
for g in range(1,10):
    svm_model = svm.SVC(kernel='rbf', C=1, gamma=g)
    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_val)
    ME = np.sum(y_pred != y_val)
    MR = ME/len(y_pred)
    ACC = 1 - MR
    print(f'\tAccuracy [gamma={g}]: {ACC}') 

# Kernel SIGMOID -> scelta di gamma
print('\n[SIGMOID] Accuracy with different gamma:')
for g in range(1,10):
    svm_model = svm.SVC(kernel='sigmoid', C=1, gamma=g)
    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_val)
    ME = np.sum(y_pred != y_val)
    MR = ME/len(y_pred)
    ACC = 1 - MR
    print(f'\tAccuracy [gamma={g}]: {ACC}')


# Kernel POLY con grado 5 -> scelta del costo
print('\n[POLY, DEGREE=5] Accuracy with different cost:')
for c in range(1,10):
    svm_model = svm.SVC(kernel='poly', C=c, degree=5)
    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_val)
    ME = np.sum(y_pred != y_val)
    MR = ME/len(y_pred)
    ACC = 1 - MR
    print(f'\tAccuracy [cost={c}]: {ACC}')
    # costo ottimale = 1
    # ACC = 60%





# VALUTAZIONE DELLE PERFORMANCE ***********************************************

# Riaddestramento del modello sull'intero set (training e validation)
#x_train = pd.concat([x_train, x_val])
#y_train = pd.concat([y_train, y_val])


# LOGISTIC REGRESSION
# Predizioni sul test set
#logreg_model.fit(x_train, y_train)         # riaddestramento
y_pred_logreg = logreg_model.predict(x_test)

# Metriche: ME, MR, ACC
ME = np.sum(y_pred_logreg != y_test)
MR = np.mean(y_pred_logreg != y_test)
ACC = 1 - MR

print('\nMetrics for Logistic Regression performance valutation:')
print(f'\tME = {ME}')
print(f'\tMR = {MR}')
print(f'\tMper = {MR*100}%')
print(f'\tACC = {ACC}')
# ACC = 57,3%


# SVM
# Predizioni sul test set
svm_model = svm.SVC(kernel='poly', C=1, degree=5)
svm_model.fit(x_train, y_train)             # fitting con iperparametri scelti
y_pred_SVM = svm_model.predict(x_test)

# Metriche: ME, MR, ACC
ME = np.sum(y_pred_SVM != y_test)
MR = np.mean(y_pred_SVM != y_test)
ACC = 1 - MR

print('\nMetrics for SVM performance valutation:')
print(f'\tME = {ME}')
print(f'\tMR = {MR}')
print(f'\tMper = {MR*100}%')
print(f'\tACC = {ACC}')
# ACC = 57,7%

# MODELLO OTTIMALE SCELTO (seed=8): SVM con kernel='poly', costo=1, grado=5

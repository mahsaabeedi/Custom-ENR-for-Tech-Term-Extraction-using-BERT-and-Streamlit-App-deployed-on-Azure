
import numpy as np 
from statistics import stdev 
colmns=['pollution','dew','temp','pressure','wind_dir','wind_spd','snow','rain']
pearso_cor=np.zeros(8)
n = 12000
file = np.load('polution_dataSet.npy')
############compute correlation

x=file[:,0]
stdx=np.std(x)   
for i in range(8):
  y=file[:,i]
  stdy=np.std(y) 
  convv=np.cov(x, y, bias=True)[0][1]
  pearso_cor[i]=convv/(stdx *stdy)
  
##########select colmns with maximum absoulut corolation value(unayi k gadre motlageshun bishtare entxab mishan)

val=np.zeros(3)
ind=np.zeros(3)
print(' pearson correlation value for each column:',pearso_cor)

for i in range(3):
  pearso_cor=np.absolute(pearso_cor)
  ind[i]=np.argmax(pearso_cor)
  val[i]=np.max(pearso_cor)
  pearso_cor[int(ind[i])]=0

print('\n3max pearson correlation absolute value:',val)
print('\n3 selected columns:',colmns[int(ind[0])],colmns[int(ind[1])],colmns[int(ind[2])],'\n')

#########copy selected columns to (data) array, colmn0:pollution, column 1:wind_spd , column2: wind_dir

data=np.zeros((len(file),3))
for i in range(len(file)):
  data[i][0] = file[i,int(ind[0])]
  data[i][1] = file[i,int(ind[1])]
  data[i][2] = file[i,int(ind[2])]

# arrary shape (43799,3)
data

#######2 ta column ro ba tabehaye feature_selection sklearn chek kardam ,unm in 2ta ro entxab mikone!!!  
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
fs = SelectKBest(score_func=f_regression, k=2)
# apply feature selection
X_selected = fs.fit_transform(file[:,1:], file[:,0])
print(X_selected.shape)
print(X_selected)


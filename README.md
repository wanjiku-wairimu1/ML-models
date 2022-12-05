# ML-models
cross validation of algorithms
# cross validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


models = []
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('LR', LinearRegression()))
models.append(('GB', GradientBoostingRegressor()))
models.append(('RFR', RandomForestRegressor()))
models.append(('SVM', SVR()))

for name, model in models:
kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = 'r2')
print(name, cv_results.mean())
# Classification
# Step 2a: Cross Validation Of Algorithms/Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
 

models = []
models.append(('KNN', KNeighborsClassifier())
models.append(('DT', DecisionTreeClassifier())
models.append(('GB', GradientBoostingClassifier()))
models.append(('RFR', RandomForestClassifier()))
models.append(('SVM', SVC()))
 

# KFOLD - Cross Validation
for name, model in models:
kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = 'accuracy')
print(name, cv_results.mean())

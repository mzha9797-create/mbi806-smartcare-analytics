from pathlib import Path
import json
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
d=pd.read_csv(BASE/'04_Data'/'smartwatch_health_data.csv')
features=[c for c in d.columns if c not in ['PatientID','WeeklyHealthScore']]
X=d[features];y=d.WeeklyHealthScore
train,test=train_test_split(d.index,test_size=.2,random_state=42,shuffle=True)
num=[c for c in features if c!='Gender']
prep=ColumnTransformer([('numeric',StandardScaler(),num),('gender',OneHotEncoder(handle_unknown='ignore',drop='if_binary'),['Gender'])])
pipe=Pipeline([('preprocess',prep),('ridge',Ridge())])
cv=KFold(n_splits=5,shuffle=True,random_state=42)
search=GridSearchCV(pipe,{'ridge__alpha':[.1,1,10,100,1000]},scoring='neg_mean_absolute_error',cv=cv,refit=True)
search.fit(X.loc[train],y.loc[train])
# One fixed split. Hyperparameter selection is training-only. Test is opened once below.
model=search.best_estimator_
baseline=DummyRegressor(strategy='mean').fit(X.loc[train],y.loc[train])
pred=model.predict(X.loc[test]);meanpred=baseline.predict(X.loc[test])
def score(actual,pred):return {'MAE':float(mean_absolute_error(actual,pred)),'RMSE':float(np.sqrt(mean_squared_error(actual,pred))),'R2':float(r2_score(actual,pred))}
out=pd.DataFrame({'PatientID':d.loc[test,'PatientID'],'observed':y.loc[test],'ridge_prediction':pred,'baseline_prediction':meanpred})
out['ridge_absolute_error']=(out.observed-out.ridge_prediction).abs();out['baseline_absolute_error']=(out.observed-out.baseline_prediction).abs()
out.to_csv(BASE/'05_Evidence'/'test_predictions.csv',index=False)
split=d[['PatientID']].copy();split['split']=np.where(split.index.isin(test),'test','train');split.to_csv(BASE/'05_Evidence'/'train_test_split.csv',index=False)
cvresults=pd.DataFrame(search.cv_results_)[['param_ridge__alpha','mean_test_score','std_test_score','rank_test_score']];cvresults['CV_MAE']=-cvresults.mean_test_score
cvresults.to_csv(BASE/'05_Evidence'/'training_cv.csv',index=False)
baseline_cv=cross_val_predict(DummyRegressor(strategy='mean'),X.loc[train],y.loc[train],cv=cv)
fitted=model.predict(X.loc[train])
rng=np.random.default_rng(42);err=(out.ridge_absolute_error-out.baseline_absolute_error).to_numpy();idx=rng.integers(0,10,(5000,10));interval=np.quantile(err[idx].mean(axis=1),[.025,.975]).tolist()
metrics={'target':'WeeklyHealthScore','features':features,'train_n':40,'test_n':10,'random_state':42,'split_type':'Random holdout; no temporal backtest possible without dates','selected_alpha':float(search.best_params_['ridge__alpha']),'selected_cv_MAE':float(-search.best_score_),'baseline_cv_MAE':float(mean_absolute_error(y.loc[train],baseline_cv)),'ridge_train':score(y.loc[train],fitted),'ridge_test':score(y.loc[test],pred),'baseline_test':score(y.loc[test],meanpred),'train_target_mean':float(y.loc[train].mean()),'test_target_mean':float(y.loc[test].mean()),'mae_difference_ridge_minus_baseline':float(err.mean()),'paired_bootstrap_difference_ci':interval,'test_ids':out.PatientID.tolist(),'interpretation':'Exploratory regression of supplied opaque score; not disease prediction. Selected CV result is not nested and can be optimistic.'}
(BASE/'05_Evidence'/'model_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
coef=pd.DataFrame({'feature':model.named_steps['preprocess'].get_feature_names_out(),'coefficient':model.named_steps['ridge'].coef_});coef.to_csv(BASE/'05_Evidence'/'ridge_coefficients.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
fig,axs=plt.subplots(1,2,figsize=(9,4));axs[0].scatter(out.observed,out.ridge_prediction,color='#177E89',s=50);axs[0].plot([50,100],[50,100],ls='--',color='grey');axs[0].set(xlim=(50,100),ylim=(50,100),xlabel='Observed supplied score',ylabel='Ridge predicted score',title='Independent test set (n = 10)')
axs[1].bar(['Mean baseline','Ridge'],[metrics['baseline_test']['MAE'],metrics['ridge_test']['MAE']],color=['#8D9AAE','#177E89']);axs[1].set(ylabel='Test MAE (score points)',title='Lower is better')
for i,v in enumerate([metrics['baseline_test']['MAE'],metrics['ridge_test']['MAE']]):axs[1].text(i,v+.15,f'{v:.2f}',ha='center')
plt.tight_layout();plt.savefig(BASE/'05_Evidence'/'figures'/'07_model_test.png',dpi=180,bbox_inches='tight');plt.close()
print(json.dumps(metrics,indent=2))

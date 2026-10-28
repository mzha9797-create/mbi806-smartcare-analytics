from pathlib import Path
import json, hashlib, itertools, platform
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'04_Data'; CODE=BASE/'03_Code'; FIG=BASE/'05_Evidence'/'figures'
for p in [DATA,CODE,FIG,BASE/'01_Reports',BASE/'02_Presentation',BASE/'06_Study_Guide',BASE/'07_Dashboard']: p.mkdir(parents=True,exist_ok=True)
source=DATA/'smartwatch_health_data.csv'
if not source.exists(): raise FileNotFoundError('Place smartwatch_health_data.csv in the sibling 04_Data folder')
raw=pd.read_csv(source)
d=raw.copy()
assert len(d)==50 and d.PatientID.nunique()==50
numeric=d.select_dtypes('number').columns.tolist()
missing=int(d.isna().sum().sum()); duplicates=int(d.duplicated().sum())
hr_bad=(d.HeartRate_Min>d.HeartRate_Avg)|(d.HeartRate_Avg>d.HeartRate_Max)
assert not hr_bad.any()
assert d.Age.ge(50).all() and d.FallAlerts.isin([0,1]).all()
for c in ['SleepQualityScore','StressIndex','WeeklyHealthScore']: assert d[c].between(0,100).all()
assert d.SleepDuration_Hours.between(0,24).all()
d['AgeGroup']=pd.cut(d.Age,[49,59,69,79],labels=['50-59','60-69','70-79']).astype(str)

def assign(frame, steps=5000, sleep=7, stress=70):
 out=frame.copy()
 out['ShortSleep']=(out.SleepDuration_Hours<sleep).astype(int)
 out['LowActivity']=(out.DailyStepCount<steps).astype(int)
 out['HighStress']=(out.StressIndex>=stress).astype(int)
 out['ReviewPoints']=2*out.FallAlerts+out.ShortSleep+out.LowActivity+out.HighStress
 out['Priority']=np.where(out.FallAlerts.eq(1)|out.ReviewPoints.ge(3),'Priority review',np.where(out.ReviewPoints.ge(1),'Routine support','No rule flag'))
 out['Reasons']=out.apply(lambda r:'; '.join(t for c,t in [('FallAlerts','Device fall alert'),('ShortSleep','Sleep below threshold'),('LowActivity','Steps below threshold'),('HighStress','Stress at or above threshold')] if r[c]) or 'No rule flag',axis=1)
 return out
d=assign(d)
# Fixed policy normalisation, not fitted clinical probabilities.
d['ActivityGap01']=(1-d.DailyStepCount/5000).clip(0,1)
d['SleepGap01']=(1-d.SleepDuration_Hours/7).clip(0,1)
d['Stress01']=d.StressIndex/100
d.to_csv(DATA/'smartwatch_prepared.csv',index=False)
desc=raw[numeric].describe().T
desc.to_csv(BASE/'05_Evidence'/'descriptive_statistics.csv')
iqr=[]
for c in numeric:
 if c=='FallAlerts': continue
 q1,q3=d[c].quantile([.25,.75]); mask=(d[c]<q1-1.5*(q3-q1))|(d[c]>q3+1.5*(q3-q1))
 iqr.append({'field':c,'count':int(mask.sum()),'lower':q1-1.5*(q3-q1),'upper':q3+1.5*(q3-q1)})
pd.DataFrame(iqr).to_csv(BASE/'05_Evidence'/'outlier_audit.csv',index=False)

rng=np.random.default_rng(STUDENT_ID_REMOVED)
rho,p=stats.spearmanr(d.SleepDuration_Hours,d.StressIndex)
rx=stats.rankdata(d.SleepDuration_Hours); ry=stats.rankdata(d.StressIndex)
perms=np.array([np.corrcoef(rx,rng.permutation(ry))[0,1] for _ in range(19999)])
perm_p=(1+np.sum(np.abs(perms)>=abs(rho)))/20000
idx=rng.integers(0,len(d),size=(5000,len(d)))
boots=np.array([stats.spearmanr(d.SleepDuration_Hours.to_numpy()[z],d.StressIndex.to_numpy()[z]).statistic for z in idx])
ci=np.quantile(boots,[.025,.975])
pairs=[]
for a,b in itertools.combinations(numeric,2):
 r,pv=stats.spearmanr(d[a],d[b]);pairs.append({'x':a,'y':b,'rho':r,'p':pv})
cor=pd.DataFrame(pairs).sort_values('p').reset_index(drop=True)
cor['q_BH']=np.minimum.accumulate((cor.p*len(cor)/np.arange(1,len(cor)+1))[::-1])[::-1].clip(upper=1)
cor.to_csv(BASE/'05_Evidence'/'correlation_tests.csv',index=False)
age=d.groupby('AgeGroup').agg(n=('Age','size'),steps=('DailyStepCount','mean'),sleep=('SleepDuration_Hours','mean'),stress=('StressIndex','mean'),falls=('FallAlerts','sum'),priority=('Priority',lambda x:int((x=='Priority review').sum()))).reset_index()
gender=d.groupby('Gender').agg(n=('Age','size'),steps=('DailyStepCount','mean'),sleep=('SleepDuration_Hours','mean'),stress=('StressIndex','mean'),priority=('Priority',lambda x:int((x=='Priority review').sum()))).reset_index()
for t in [age,gender]: t['priority_rate']=t.priority/t.n
age.to_csv(BASE/'05_Evidence'/'age_summary.csv',index=False);gender.to_csv(BASE/'05_Evidence'/'gender_summary.csv',index=False)
def wilson(k,n):
 z=1.95996398454; p=k/n; den=1+z*z/n
 return [(p+z*z/2/n-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den,(p+z*z/2/n+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den]
scenarios=[]
baseline=set(d.loc[d.Priority.eq('Priority review'),'PatientID'])
for step,sl,st in itertools.product([4000,5000,6000],[6.5,7,7.5],[65,70,75]):
 tmp=assign(raw,step,sl,st); s=set(tmp.loc[tmp.Priority.eq('Priority review'),'PatientID'])
 scenarios.append({'steps':step,'sleep':sl,'stress':st,'priority':len(s),'routine':int(tmp.Priority.eq('Routine support').sum()),'unflagged':int(tmp.Priority.eq('No rule flag').sum()),'jaccard':len(s&baseline)/len(s|baseline)})
sens=pd.DataFrame(scenarios);sens.to_csv(BASE/'05_Evidence'/'sensitivity.csv',index=False)
metrics={'n':len(d),'columns':len(raw.columns),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'missing_cells':missing,'duplicate_rows':duplicates,'hr_violations':int(hr_bad.sum()),'outliers':iqr,'means':raw[numeric].mean().to_dict(),'medians':raw[numeric].median().to_dict(),'counts':d.Priority.value_counts().to_dict(),'flags':d[['FallAlerts','ShortSleep','LowActivity','HighStress']].sum().astype(int).to_dict(),'sleep_stress':{'rho':rho,'permutation_p':perm_p,'bootstrap_ci':ci.tolist(),'bootstrap_replicates':5000,'permutations':19999},'correlations_tested':len(cor),'min_q':float(cor.q_BH.min()),'significant_q05':int(cor.q_BH.lt(.05).sum()),'age':age.to_dict('records'),'gender':gender.to_dict('records'),'priority_wilson':wilson(8,50),'fall_wilson':wilson(5,50),'sensitivity_priority_range':[int(sens.priority.min()),int(sens.priority.max())],'sensitivity_jaccard_range':[float(sens.jaccard.min()),float(sens.jaccard.max())],'seed':STUDENT_ID_REMOVED,'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__}}
(BASE/'05_Evidence'/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
sns.set_theme(style='whitegrid',font='DejaVu Sans',font_scale=1.05)
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.bbox':'tight'})
teal='#177E89'; amber='#B7791F'; blue='#3155A6'
def save(name):
 plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=180);plt.close()
fig,ax=plt.subplots(figsize=(8.7,4.1)); names=['Sleep <7 h','Stress >=70','Steps <5,000','Fall alert']; vals=[27,21,12,5]
ax.barh(names[::-1],vals[::-1],color=teal);ax.set_xlim(0,50);ax.set_xlabel('Patients out of 50 (flags can overlap)')
for i,v in enumerate(vals[::-1]):ax.text(v+.6,i,f'{v} ({v/50:.0%})',va='center')
save('01_flags')
fig,ax=plt.subplots(figsize=(8.7,4.6));sns.scatterplot(data=d,x='SleepDuration_Hours',y='StressIndex',hue='Gender',palette=[teal,blue],s=65,ax=ax)
ax.axvline(7,color='grey',ls='--',lw=1);ax.axhline(70,color='grey',ls='--',lw=1);ax.set(xlabel='Sleep duration (hours)',ylabel='Supplied stress index (0 to 100)',title=f'Sleep and stress association   Spearman rho = {rho:.3f}')
save('02_sleep_stress')
fig,axs=plt.subplots(1,2,figsize=(9,4));axs[0].bar(age.AgeGroup,age.steps,color=teal);axs[0].set(ylabel='Mean daily steps',ylim=(0,10000));axs[1].bar(age.AgeGroup,age.sleep,color=blue);axs[1].set(ylabel='Mean sleep (hours)',ylim=(0,10))
for ax in axs:
 ax.set_xlabel('Age group (years)')
 for i,n in enumerate(age.n): ax.text(i,ax.get_ylim()[1]*.95,f'n={n}',ha='center')
save('03_age_groups')
cols=['DailyStepCount','HeartRate_Avg','SleepDuration_Hours','SleepQualityScore','StressIndex','WeeklyHealthScore']
labels=['Steps','Mean HR','Sleep hours','Sleep quality','Stress','Health score']
fig,ax=plt.subplots(figsize=(8.3,5));sns.heatmap(d[cols].corr(method='spearman'),vmin=-1,vmax=1,cmap='vlag',annot=True,fmt='.2f',xticklabels=labels,yticklabels=labels,ax=ax,cbar_kws={'label':'Spearman rho'});save('04_heatmap')
fig,axs=plt.subplots(1,2,figsize=(9,4));v=[8,33,9];axs[0].pie(v,labels=['Priority\n8','Routine\n33','No flag\n9'],colors=[amber,teal,'#D7E1EA'],startangle=90,wedgeprops={'width':.42});axs[0].text(0,0,'50\npatients',ha='center',va='center')
rates=gender.priority_rate.to_numpy();cis=np.array([wilson(int(k),int(n)) for k,n in zip(gender.priority,gender.n)])
axs[1].bar(gender.Gender,rates*100,color=[teal,blue]);axs[1].errorbar(range(2),rates*100,yerr=np.vstack([rates-cis[:,0],cis[:,1]-rates])*100,fmt='none',ecolor='black',capsize=5);axs[1].set(ylabel='Priority review rate (%)',ylim=(0,55),title='Wilson 95% intervals')
save('05_priority')
fig,ax=plt.subplots(figsize=(8.7,4));grid=sens[sens.stress.eq(70)].pivot(index='sleep',columns='steps',values='priority');sns.heatmap(grid,annot=True,fmt='d',cmap='YlGnBu',ax=ax,cbar_kws={'label':'Patients in priority review'});ax.set(xlabel='Step threshold (<)',ylabel='Sleep threshold in hours (<)',title='Policy sensitivity with stress threshold fixed at 70');save('06_sensitivity')
print(json.dumps(metrics,indent=2))

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parents[1]
data=base/"data";out=base/"visuals";out.mkdir(exist_ok=True)
def save(name):
 plt.tight_layout();plt.savefig(out/name,dpi=150,bbox_inches="tight");plt.close()
d=pd.read_csv(data/"review_allocation.csv");m=pd.read_csv(data/"model_summary.csv")
fig,axs=plt.subplots(1,2,figsize=(11,4.5));axs[0].bar(d.Category,d.Patients,color=["#B47A22","#147D92","#8BA4B6"]);axs[0].set(title="Illustrative review allocation",ylabel="Records",ylim=(0,40));axs[0].tick_params(axis="x",rotation=15)
axs[1].bar(m.Model,m.MAE,color=["#147D92","#3769B0"]);axs[1].set(title="Independent test: 10 records",ylabel="MAE (supplied score points)",ylim=(0,16));axs[1].tick_params(axis="x",rotation=15)
fig.suptitle("SmartCare: aggregate educational results; human review required");save("overview.png")
d=pd.read_csv(data/"age_group_summary.csv");fig,axs=plt.subplots(1,2,figsize=(11,4.5));axs[0].bar(d.AgeGroup,d.MeanSteps,color="#147D92");axs[0].set(title="Mean steps by age band",ylabel="Steps/day");axs[1].bar(d.AgeGroup,d.MeanSleepHours,color="#3769B0");axs[1].set(title="Mean sleep by age band",ylabel="Hours",ylim=(0,9));save("age_groups.png")

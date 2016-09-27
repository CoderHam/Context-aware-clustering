import pandas as pd

df_cols = pd.read_csv("cols",header=None)
cols = df_cols[0].values.tolist()
df_flags = pd.read_csv("flag.data",header=None,names=cols)
rel = pd.read_csv("religion",header=None)[0].values.tolist()
landm = pd.read_csv("landmass",header=None)[0].values.tolist()
lang = pd.read_csv("language",header=None)[0].values.tolist()
zone = pd.read_csv("zone",header=None)[0].values.tolist()

df_flags["religion"].replace([x for x in xrange(8)],rel,True)
df_flags["landmass"].replace([x for x in xrange(1,len(landm)+1)],landm,True)
df_flags["language"].replace([x for x in xrange(1,len(lang)+1)],lang,True)
df_flags["zone"].replace([x for x in xrange(1,len(zone)+1)],zone,True)

colors = ['red','green','blue','gold','white','black','orange']

for col in colors:
	df_flags[col].replace([1,0],[col,""],True)
	df_flags["color"] += df_flags[col]

print df_flags[:2]
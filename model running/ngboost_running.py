from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
import time
from sklearn.tree import DecisionTreeRegressor

from hadden_theory.test_particle_secular_hamiltonian import SyntheticSecularTheory, TestParticleSecularHamiltonian, calc_g0_and_s0
from hadden_theory import test_particle_secular_hamiltonian
# hack to make pickle load work
import sys
sys.modules['test_particle_secular_hamiltonian'] = test_particle_secular_hamiltonian

try:
	plt.style.use('/Users/dtamayo/.matplotlib/paper.mplstyle')
except:
	pass

from pathlib import Path
Path("tables_for_analysis").mkdir(exist_ok=True)

# Read Nesvorny catalog dataset
nesvorny_df = pd.read_csv("nesvorny_catalog_dataset.csv")

# Read linear prediction results
prediction_path = Path("linear_predictions")
file_names = list(prediction_path.glob("*.npz"))
rows = []

for i in range(len(file_names)):
	soln_h = np.load(file_names[i])
	prope_value = soln_h["u"]
	propsini_value = soln_h["v"]
	g0_value = soln_h["g"]
	s0_value = soln_h["s"]
	des_n = file_names[i].stem.replace("integration_results_", "")
	rows.append([des_n, prope_value, propsini_value, g0_value, s0_value])

df_h = pd.DataFrame(rows, columns=["Des'n", "prope_h", "propsini_h", "g0", "s0"])

# Get merged dataframe for later model training
merged_df = pd.merge(nesvorny_df, df_h, on="Des'n", how="inner")

features = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'propa', 's0', 'propsini_h']
data = merged_df[features]
dela = merged_df['propa']-merged_df['a']
dele = merged_df['prope']-merged_df['e']
delsini = merged_df['propsini']-np.sin(merged_df['Incl.']*np.pi/180)
delg = merged_df['g0'] - merged_df['g']
s = merged_df['s']

trainX, testX, trainY, testY = train_test_split(data, delsini, test_size=0.2, random_state=42)
valX, testX, valY, testY = train_test_split(testX, testY, test_size=0.5, random_state=42)
space = {
	'max_depth': hp.qloguniform('x_max_depth', np.log(5), np.log(40), 1),
	'minibatch_frac': hp.uniform('minibatch_frac', 0.1, 1.0),
	'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3))
}

def objective(params):
	clf = NGBRegressor(
		Dist=Normal,
		Score=LogScore,
		verbose=False,
		minibatch_frac=params['minibatch_frac'],
		n_estimators=200,
		learning_rate=params['learning_rate'],
		Base=DecisionTreeRegressor(max_depth=int(params['max_depth']))
	)

	clf.fit(trainX, trainY)    
	preds = clf.pred_dist(valX)
	mu = preds.loc
	rmse = np.sqrt(np.mean((valY-mu)**2))

	return {'loss': rmse, 'status': STATUS_OK}

trials = Trials()
start = time.time()

best = fmin(fn=objective, space = space, algo = tpe.suggest, max_evals = 10, trials = trials, rstate=np.random.default_rng(seed=0))
end = time.time()
print("Best hyperparameters:", best)
print("Optimization Time: %.2f seconds" % (end - start))

final_model = NGBRegressor(
	# Dist=Normal,
	# Score=LogScore,
	# n_estimators=500,
	# natural_gradient=True,
	minibatch_frac= 0.7325219177053108, # 1
	learning_rate=0.040346736610987484, # 0.1
	Base=DecisionTreeRegressor(max_depth = 13) # 6
)

final_model.fit(trainX, trainY)

ngb = Path("models/best_model_inc_val_2.ngb")
with ngb.open("wb") as f:
    pickle.dump(final_model, f)
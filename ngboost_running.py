# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import ngboost
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import hyperopt

from hadden_theory.test_particle_secular_hamiltonian import SyntheticSecularTheory, TestParticleSecularHamiltonian, calc_g0_and_s0
from hadden_theory import test_particle_secular_hamiltonian
# hack to make pickle load work
import sys
sys.modules['test_particle_secular_hamiltonian'] = test_particle_secular_hamiltonian

try:
	plt.style.use('/Users/dtamayo/.matplotlib/paper.mplstyle')
except:
	pass
# %%
from pathlib import Path
Path("tables_for_analysis").mkdir(exist_ok=True)

merged_df = pd.read_csv("merged_elements.csv")

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
# %%
features = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'propa', 'g0', 'prope_h']
data = merged_df[features]
dela = merged_df['propa']-merged_df['a']
dele = merged_df['prope']-merged_df['e']
delsini = merged_df['propsini']-np.sin(merged_df['Incl.']*np.pi/180)
delg = merged_df['g0'] - merged_df['g']
s = merged_df['s']

trainX, testX, trainY, testY = train_test_split(data, dele, test_size=0.4, random_state=42)
valX, testX, valY, testY = train_test_split(testX, testY, test_size=0.5, random_state=42)
space = {
	'max_depth': hp.qloguniform('max_depth', np.log(5), np.log(40), 1),
	'minibatch_frac': hp.uniform('minibatch_frac', 0.1, 1.0),
	'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3))
}
# %%
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

best = fmin(fn=objective, space = space, algo = tpe.suggest, max_evals = 15, trials = trials, rstate=np.random.default_rng(seed=0))
end = time.time()
print("Best hyperparameters:", best)
print("Optimization Time: %.2f seconds" % (end - start))
# %%
# best = {'minibatch_frac': 0.4670652921193636, 'learning_rate': 0.04872472351951210, 'max_depth': 14.0}
final_model = NGBRegressor(
	# Dist=Normal,
	# Score=LogScore,
	# n_estimators=500,
	# natural_gradient=True,
	minibatch_frac=best['minibatch_frac'],
	learning_rate=best['learning_rate'],
	Base=DecisionTreeRegressor(max_depth=int(best['max_depth']))
)

final_model.fit(trainX, trainY)

ngb = Path("models/best_model_ecc_val_2.ngb")
with ngb.open("wb") as f:
    pickle.dump(final_model, f)
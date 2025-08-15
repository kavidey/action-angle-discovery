# %%
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
# %%
from pathlib import Path
Path("tables_for_analysis").mkdir(exist_ok=True)

merged_df = pd.read_csv("merged_elements.csv")
# %%
features_e = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'propa', 'g0', 's0', 'prope_h', 'propsini_h']
features_inc = ['sinicosO', 'sinisinO', 'ecospo', 'esinpo', 'propa', 'g0', 's0', 'prope_h', 'propsini_h']
features = features_inc

data = merged_df[features]
dele = merged_df['prope']-merged_df['e']
delsini = merged_df['propsini']-np.sin(merged_df['Incl.']*np.pi/180)

trainX, testX, trainY, testY = train_test_split(data, delsini, test_size=0.4, random_state=42)
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
best_ecc = {'minibatch_frac': 0.497938840723922, 'learning_rate': 0.0403467366109875, 'max_depth': 22.0}
best_inc = {'minibatch_frac': 0.4670652921193636, 'learning_rate': 0.04872472351951210, 'max_depth': 14.0}
best = best_inc
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

ngb = Path("models/best_model_inc_val_2.ngb")
with ngb.open("wb") as f:
    pickle.dump(final_model, f)
# plot decision tree
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
##set up  the parameters for image frame size
rcParams['figure.figsize'] = 80,50

# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot single tree
plot_tree(model)


plot_tree(model, num_trees=4)
	
plot_tree(model, num_trees=0, rankdir='LR')

plt.show()
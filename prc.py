from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
def plot_AUCPRC(labels, predictions):
	"""labels: a 1d list of all the labels (0 or 1) for our samples
		predictions: a 1d list of all the probabilities (between 0 and 1)
		             for each one of our samples. It should be the probability that
		             that this sample belongs to the second class (e.g a splice site.)
	"""
	# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument\
		
	precision, recall, _ = precision_recall_curve(labels, predictions)
	lr_auc = auc(recall, precision)
	step_kwargs = ({'step': 'post'}
	               if 'step' in signature(plt.fill_between).parameters
	               else {})
	plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: auc={0:0.4f}'.format(
	          lr_auc))
	plt.savefig("/home/brian/bert_6_split_1.png")

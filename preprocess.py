import re
def get_data(filepath):
	vocab = set()
	genes = []
	labels = []
	max_len = 0
	count = 0
	with open(filepath, 'r') as f:
		l = f.readline()
		l = f.readline()
		while l != None:
			inside_bracks = re.findall(r'\[(.*?)\]',l)
			if len(inside_bracks) < 2:
				break
			if count == 300:
				break
			mers = inside_bracks[0].split(", ")
			mers = [mer[1:-1] for mer in mers]
			if len(mers) > max_len:
				max_len = len(mers)
			genes.append(mers)
			gene_labels = inside_bracks[1].split(", ")
			gene_labels = list(map(float, gene_labels)) 
			labels.append(gene_labels)
			vocab.update(mers)
			l = f.readline()
			count += 1
	return vocab, genes, labels, max_len

def pad(genes, max_len):
	atten_masks = []
	for i in range(len(genes)):
		length = len(genes[i])
		diff = max_len - length
		genes[i].extend(['[PAD]' for i in range(diff)])
		mask = [1 for j in range(length)]
		mask.extend([0 for j in range(diff)])
		atten_masks.append(np.array(mask))
	return genes, np.array(atten_masks)

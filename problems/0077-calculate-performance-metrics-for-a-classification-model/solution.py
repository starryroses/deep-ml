
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	confusion_matrix = [[0 for i in range(2)]for y in range(2)]
	for i in range(len(actual)):
		if actual[i]==1 and predicted[i]==1:     #tp
			confusion_matrix[0][0] +=1
		if actual[i]==1 and predicted[i]==0:     #fn
			confusion_matrix[0][1]+=1
		if actual[i]==0 and predicted[i]==1:     #fp
			confusion_matrix[1][0]+=1
		if actual[i]==0 and predicted[i]==0:     #tn
			confusion_matrix[1][1]+=1	
	
	accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1])/(confusion_matrix[1][1] + confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[0][1])
	specificity = confusion_matrix[1][1] / (confusion_matrix[1][1]+confusion_matrix[1][0])
	negativePredictive = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
	precision = confusion_matrix[0][0] / (confusion_matrix[0][0]+confusion_matrix[1][0])
	recall = confusion_matrix[0][0] / (confusion_matrix[0][0]+confusion_matrix[0][1])
	f1 = 2*precision*recall / (precision+recall)
	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)

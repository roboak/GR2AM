from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
true_labels = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,11,11,11,12,12,12,13,13,13,14,14,14,15,15,15]
predicted_labels = [2,2,2,2,2,3,3,3,3,4,4,4,5,5,6,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,11,11,11,12,12,12,13,13,13,14,14,14,15,15,15]
display_labels_1 = list(set(true_labels))
confusion_matrix_1 = confusion_matrix(true_labels, predicted_labels)
print(f"Accuracy {accuracy_score(true_labels, predicted_labels)} ")
print("Participant1 ML 1")
# fig = plt.figure()
plt.figure(figsize=(8, 6))
# plt.imshow(confusion_matrix_1, interpolation='nearest')
display_1 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_1, display_labels=display_labels_1)
display_1.plot()
# plt.plot()
# plt.savefig('saved_figure.png')
# plt.show()
fig = plt.gcf()
fig.savefig('saved_figure.png')


import matplotlib.pyplot as plt
import numpy as np

accL = [84.9, 84.12, 88.9, 89.22]
eerL = [17.56, 17.93, 15.58, 15.42]
treen = [2, 3, 4, 5]

plt.figure(figsize=(12, 5))

# make columns equidistant (for HMOG plots)
x_values = np.arange(len(treen))

plt.subplot(1, 2, 1)
plt.bar(x_values, accL, color='c', alpha=0.7, label='Accuracy (%)', width=0.4)
plt.xticks(x_values, treen)  # Set x-axis labels
plt.title('Accuratezza vs tempo di autenticazione')
plt.xlabel('t')
plt.ylabel('Accuratezza (%)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(x_values, eerL, color='g', alpha=0.7, label='EER (%)', width=0.4)
plt.xticks(x_values, treen)  # Set x-axis labels
plt.title('EER vs tempo di autenticazione')
plt.xlabel('t')
plt.ylabel('EER (%)')
plt.grid(True)

plt.suptitle('OneClass Support Vector Machine - BrainRun')
plt.tight_layout()
plt.show()

accL = [72, 84.19, 87.69, 88.48]
eerL = [16.74, 17.37, 17.9, 28.6]
treen = [0.2, 0.5, 1, 2]

plt.figure(figsize=(12, 5))

x_values = np.arange(len(treen))

plt.subplot(1, 2, 1)
plt.bar(x_values, accL, color='c', alpha=0.7, label='Accuracy (%)', width=0.4)
plt.xticks(x_values, treen)  # Set x-axis labels
plt.title('Accuratezza vs tempo di autenticazione')
plt.xlabel('t')
plt.ylabel('Accuratezza (%)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(x_values, eerL, color='g', alpha=0.7, label='EER (%)', width=0.4)
plt.xticks(x_values, treen)  # Set x-axis labels
plt.title('EER vs tempo di autenticazione')
plt.xlabel('t')
plt.ylabel('EER (%)')
plt.grid(True)

plt.suptitle('OneClass Support Vector Machine - HMOG')
plt.tight_layout()
plt.show()

accL = [89.68, 91.63, 92.1, 91.21]
eerL = [17.68, 16.68, 16.52, 17.1]
treen = [2, 3, 4, 5]

plt.figure(figsize=(12, 5))

x_values = np.arange(len(treen))

plt.subplot(1, 2, 1)
plt.bar(x_values, accL, color='c', alpha=0.7, label='Accuracy (%)', width=0.4)
plt.xticks(x_values, treen)  # Set x-axis labels
plt.title('Accuratezza vs tempo di autenticazione')
plt.xlabel('t')
plt.ylabel('Accuratezza (%)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(x_values, eerL, color='g', alpha=0.7, label='EER (%)', width=0.4)
plt.xticks(x_values, treen)  # Set x-axis labels
plt.title('EER vs tempo di autenticazione')
plt.xlabel('t')
plt.ylabel('EER (%)')
plt.grid(True)

plt.suptitle('Isolation Forest - BrainRun')
plt.tight_layout()
plt.show()

accL = [59.79, 69.54, 76.1, 80.75]
eerL = [42.67, 32.7, 29.5, 27]
treen = [0.2, 0.5, 1, 2]

plt.figure(figsize=(12, 5))

x_values = np.arange(len(treen))

plt.subplot(1, 2, 1)
plt.bar(x_values, accL, color='c', alpha=0.7, label='Accuracy (%)', width=0.4)
plt.xticks(x_values, treen)  # Set x-axis labels
plt.title('Accuratezza vs tempo di autenticazione')
plt.xlabel('t')
plt.ylabel('Accuratezza (%)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(x_values, eerL, color='g', alpha=0.7, label='EER (%)', width=0.4)
plt.xticks(x_values, treen)  # Set x-axis labels
plt.title('EER vs tempo di autenticazione')
plt.xlabel('t')
plt.ylabel('EER (%)')
plt.grid(True)

plt.suptitle('Isolation Forest - HMOG')
plt.tight_layout()
plt.show()

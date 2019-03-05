import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt

# turns off interactive mode for pyplot
plt.ioff()

def plotLosses(lossArrays, lossLabels, title, filePath):
    plt.figure()
    for loss, label in zip(lossArrays, lossLabels):
        plt.plot(loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.savefig(filePath)
    plt.close()
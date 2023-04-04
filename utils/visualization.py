from matplotlib import pyplot as plt

def show_plots(history):
    """
    
    """
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    plt.xlabel("Epochs")
    ax1.set_title("Loss")

    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='vadiation')
    ax1.legend()

    ax2.set_title("Accuracy")
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='validation')
    ax2.legend()


import numpy as np
from ..data import Dataset

def show_accuracy_loss(net, scaling="scaled", test_dataset_path="../data/processed/extended"):
    """
    """
    loss = []
    accuracy = []

    for fold in [5, 6, 7, 8, 9, 10]:
        td = Dataset(dataset_path = f"{test_dataset_path}/test_{fold}_{scaling}.csv", test_size = 0)
        x_test, y_test = td.get_splits()
        results = net.evaluate(x_test, y_test, batch_size=128)
        loss.appendd(results[0])
        accuracy.append(results[1])


    print("\nAccuracy:")
    print(f"\tMean: {np.mean(accuracy)} \n\tStandard deviation: {np.std(accuracy)}")

    print("\nLoss:")
    print(f"\tMean: {np.mean(loss) \n\tStandard deviation: {np.std(loss)}")

    return accuracy, loss


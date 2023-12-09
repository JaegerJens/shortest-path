import numpy as np
import matplotlib.pyplot as plt


def plotPath(lat: np.ndarray, lng: np.ndarray) -> None:
    plt.plot(lat, lng)
    plt.show()


def plotCities(lat: np.ndarray, lng: np.ndarray,
               states: np.ndarray, pops: np.ndarray) -> None:
    citySize = pops/max(pops) * 100 + 5

    colors = ('#e6194B', '#3cb44b', '#ffe119', '#4363d8',
              '#f58231', '#911eb4', '#42d4f4', '#f032e6',
              '#bfef45', '#fabed4', '#469990', '#dcbeff',
              '#9A6324', '#fffac8', '#800000', '#aaffc3')

    federalStates = np.unique(states)
    stateColorMapping = dict(zip(federalStates, colors))
    plotColor = [stateColorMapping[s] for s in states]

    plt.scatter(lat, lng, s=citySize, c=plotColor)
    plt.show()

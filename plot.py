import os
import re

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    result_txt = "/home/dehao/github_projects/pro/results/results20221221-101357_withoutmask50_10(1).txt"
    with open(result_txt, "r") as f:
        result_str = f.read()
    print(result_str)
    IoUs = re.findall("IoU: \['(.*?)', '(.*?)'\]", result_str)
    IoUs = [[float(i[0]), float(i[1])] for i in IoUs]
    foreground_IoUs = [i[0] for i in IoUs]
    background_IoUs = [i[1] for i in IoUs]
    mean_IoUs = re.findall("mean IoU: (.*?)\n", result_str)
    mean_IoUs = [float(i) for i in mean_IoUs]
    dices = re.findall("dice coefficient: (.*?)\n", result_str)
    dices = [float(i) * 100 for i in dices]
    print(IoUs)
    print(mean_IoUs)
    print(dices)

    fig, axe = plt.subplots(1, 1, figsize=(12, 9))
    index = np.arange(len(mean_IoUs))
    axe.plot(index, mean_IoUs, label="mean IoU")
    axe.plot(index, foreground_IoUs, label="Foreground IoU")
    axe.plot(index, background_IoUs, label="Background IoU")
    axe.plot(index, dices, label="Dice")
    axe.set_xlabel("Epoch", fontsize=18)
    axe.set_ylabel("%", fontsize=18)
    axe.set_ylim(50, 100)
    axe.tick_params(labelsize=14)
    axe.grid(axis="y")
    axe.legend(loc="lower right", fontsize=18)
    fig.savefig("test.png")

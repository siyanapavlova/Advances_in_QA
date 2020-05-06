"""
This script is for plotting run, p, r, f1, and acc of the PS training
"""

import matplotlib.pyplot as plt

project_dir = "/home/simon/Desktop/LCT_Saarbrücken/Courses/AQA/project_AQA/"
outputs_dir = "models/outputs/"
filepath = project_dir + outputs_dir + "PS_final_2020-05-05.devscores"


with open(filepath, "r") as f:
     scores = []
     for line in f.readlines():
        scores.append( [float(v) for v in line.rstrip().split()])
     run, prec, rec, f1, acc = list(zip(*scores))


plot_p = plt.plot(prec, label="precision")
plot_r = plt.plot(rec, label="recall")
plot_f = plt.plot(f1, label="F1-score")
plot_a = plt.plot(acc, label="accuracy")
plt.legend()
plt.xlabel('evaluation step')
plt.show()
#print("Saving plot as", filename, "...")
#plt.savefig(filename)
#plt.close()



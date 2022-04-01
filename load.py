import numpy as np

PATH = "predict.txt"
with open(PATH, mode = 'w') as f:
    pass
a1 = np.loadtxt('result_1/predict.txt')
a2 = np.loadtxt('result_2/predict.txt')
a3 = np.loadtxt('result_3/predict.txt')

a = (a1 + a2 + a3) / 3.0

a = a*100
sq = np.sqrt((((a1*100-a)**2)+((a2*100-a)**2)+((a3*100-a)**2))/3.0)
with open(PATH, mode = 'a') as f:
    f.write("\t%.2f\t%.2f\t%.2f\n" % (a[0,0], a[0,1], a[0,2]))
with open(PATH, mode = 'a') as f:
    f.write("\t%.2f\t%.2f\t%.2f\n" % (sq[0,0], sq[0,1], sq[0,2]))
with open(PATH, mode = 'a') as f:
    f.write("\t%.2f\t%.2f\t%.2f\n" % (a[1,0], a[1,1], a[1,2]))
with open(PATH, mode = 'a') as f:
    f.write("\t%.2f\t%.2f\t%.2f\n" % (sq[1,0], sq[1,1], sq[1,2]))
print(a)

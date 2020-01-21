import re
import matplotlib.pyplot as plt


lambda_loss = []
model_loss=[]
colour_loss=[]

with open ('weights/loss', 'rt') as lossfile:
    for line in lossfile:
        lc=[m.start() for m in re.finditer('lambda_1_loss: ', line)]
        for i in lc:
            start = i+15
            lambda_loss.append(float(line[start:start + 6]))

        mc=[m.start() for m in re.finditer('predict_model_loss: ', line)]
        for j in mc:
            start = j+20
            model_loss.append(float(line[start:start + 6]))

        cc=[m.start() for m in re.finditer('predict_colour_loss: ', line)]
        for k in cc:
            start = k+21
            colour_loss.append(float(line[start:start + 6]))



print ('ok')
plt.plot(lambda_loss,  label='lamdda loss')
plt.plot(model_loss,label='model loss')
plt.plot(colour_loss, label='colour loss')
plt.legend()
plt.show()
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

dfa = pd.read_csv("./data/baseball.csv", encoding="shift-jis")
dfa.head()

dfa.rename(columns=
{'勝率': 'win_rate', '得点平均': 'score_avg', '失点平均': 'conced_avg', '打率': 'batting_avg',
 '本塁打平均': 'homerun_avg', '盗塁平均': 'stolenbase_avg', '防御率': 'earnedrun_avg'},
 inplace=True)
print("得点", np.corrcoef(dfa["score_avg"], dfa["win_rate"]))
print("失点", np.corrcoef(dfa["conced_avg"], dfa["win_rate"]))
print("打率", np.corrcoef(dfa["batting_avg"], dfa["win_rate"]))
print("本塁打", np.corrcoef(dfa["homerun_avg"], dfa["win_rate"]))
print("盗塁", np.corrcoef(dfa["stolenbase_avg"], dfa["win_rate"]))
print("防御率", np.corrcoef(dfa["earnedrun_avg"], dfa["win_rate"]))
x = dfa[['score_avg', 'conced_avg', 'batting_avg', 'homerun_avg', 'stolenbase_avg', 'earnedrun_avg']].copy()
y = dfa['win_rate']

model = linear_model.LinearRegression()
model.fit(x, y)

print('回帰係数:', model.coef_)
print('切片　:', model.intercept_)
print('決定係数:', model.score(x, y))
predict = model.predict(x)
plt.plot(np.linspace(min(y),max(y)),
         np.linspace(min(y),max(y)))
plt.plot(y, predict, 'o')

plt.xlabel('y')
plt.ylabel('predict(y)')
plt.show()
club = ["阪神","巨人","ヤクルト","中日","DeNA","広島"]
data =[
    [3.1,3.22,0.222,0.43,0.92,3.11],
    [3.02,3.76,0.247,1.25,0.54,3.02],
    [3.36,3.88,0.255,1.01,0.58,3.04],
    [4.87,3.22,0.239,0.51,0.5,3.9],
    [2.98,4.5,0.258,1.0,0.21,2.27],
    [2.78,4.2,0.26,0.84,0.48,2.8]
    ]
win = []

for i in range(6):
    predict_test = {"score_avg":[data[i][0]],"conced_avg":[data[i][1]],"batting_avg":[data[i][2]],"homerun_avg":[data[i][3]],
              "stolenbase_avg":[data[i][4]],"earnedrun_avg":[data[i][5]]}
    df_predict = pd.DataFrame(predict_test)
    xx = df_predict.iloc[:,0:6]
    yy = model.predict(xx)
    rate = yy[0]
    rate = round(rate,3)
    win.append(rate)
   
rank = pd.Series(win, index = club)

rank = rank.sort_values(ascending=False)
print(rank)
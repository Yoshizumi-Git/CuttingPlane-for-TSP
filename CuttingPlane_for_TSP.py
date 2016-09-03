#encoding=utf-8
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pulp

#データの読み込み
df=pd.read_csv("data.csv")

#点のデータと回る順番を入力して描画
def draw(df,root):
    x,y = list(df["x"]),list(df["y"])
    plt.scatter(x,y)
    for i in range(len(root)):
        plt.plot([x[i-1],x[i]],[y[i-1],y[i]],c="r")
    plt.show()

#点と枝を入力して描画
def draw_edge(df,edges):
    x,y = list(df["x"]),list(df["y"])
    plt.scatter(x,y)
    for i,j in edges:
        plt.plot([x[i],x[j]],[y[i],y[j]],c="r")
    plt.show()

#切除平面の追加
def add_cut(df,edges,problem,var):
    G = nx.Graph()
    for i,j in edges:
        G.add_edge(i,j)
    List = sorted(nx.connected_components(G), key = len, reverse=True)

    #全部連結なら終了
    if len(List)==1:
        return True

    #切除平面の作成
    temp=[[0] for l in List]
    for i,j in edges:
        for l in range(len(List)):
            if i in List[l] and j in List[l]:
                temp[l] += var[i][j]
                continue

    #切除平面の追加
    for l in range(len(List)):
        problem += temp[l] <= len(List[l])-1
    return False

#最小化問題
problem = pulp.LpProblem(sense=pulp.LpMinimize)

#変数の定義(var[i][j])
var = pulp.LpVariable.dicts('x', ([i for i in range(len(df))], [j for j in range(len(df))]), 0, 1, 'Binary')

#定数の定義
dist = {}
for i in range(len(df)):
    dist[i]={}
    for j in range(len(df)):
        dist[i][j] = np.linalg.norm(df.ix[i]-df.ix[j])

#目的関数
temp=0
for i in range(len(df)):
    for j in range(len(df)):
        temp += dist[i][j]*var[i][j]
problem += temp

#制約条件
#必ずどこかへ行く
for i in range(len(df)):
    temp = 0
    for j in range(len(df)):
        if i!=j:
            temp += var[i][j]
    problem += temp == 1

#必ずどこかから来る
for i in range(len(df)):
    temp = 0
    for j in range(len(df)):
        if i!=j:
            temp += var[j][i]
    problem += temp == 1

#切除平面を追加していくループ
while True:
    #解く
    status = problem.solve()
    print("Status", pulp.LpStatus[status])

    #枝を取って来る
    edges=[]
    for i in range(len(df)):
        for j in range(len(df)):
            if var[i][j].value() == 1.0:
                edges.append((i,j))

    #描画
    draw_edge(df,edges)
    if add_cut(df,edges,problem,var):
        break

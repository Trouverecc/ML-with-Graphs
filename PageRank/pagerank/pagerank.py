#dataset from: http://old.openkg.cn/dataset/ch4masterpieces
#西游记
import networkx as nx
import numpy as np
import random
import pandas as pd

#可视化
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.sans-serif']=['Simhei'] #中文
plt.rcParams['axes.unicode_minus']=False  #负号

#导入csv文件定义的有向图
df=pd.read_csv('data/西游记/triples.csv')
#df=pd.read_csv('data/三国演义/triples.csv')
edges=[edge for edge in zip(df['head'],df['tail'])]

G=nx.DiGraph()
G.add_edges_from(edges)
print(G)

#可视化
plt.figure(figsize=(15,14))
pos=nx.spring_layout(G,iterations=3,seed=5)
nx.draw(G,pos,with_labels=True)
plt.show()

#计算每个节点的PageRank重要度
pagerank=nx.pagerank(G,#networkx graph有向图，如果是无向图则自动转为双向有向图
                     alpha=0.85,#damping factor
                     personalization=None,#是否开启personalized PageRank，随机传送至指定节点集合的概率更高或更低
                     max_iter=100,#最大迭代次数
                     tol=1e-06,#判定收敛的误差
                     nstart=None,#每个节点初始PageRank值
                     dangling=None,)#dead end死胡同节点
#按PageRank值从高到低排序
sorted(pagerank.items(),key=lambda  x : x[1],reverse=True)

#用节点尺寸可视化PageRank值
#节点尺寸
node_sizes=(np.array(list(pagerank.values()))*8000).astype(int)
#节点颜色
M=G.number_of_edges()
edge_colors = range(2,M+2)

plt.figure(figsize=(15,14))
#绘制节点及连接
nodes=nx.draw_networkx_nodes(G,pos,node_size=node_sizes,node_color=node_sizes)
edges=nx.draw_networkx_edges(
    G,
    pos,
    node_size=node_sizes,
    arrowstyle="->",
    arrowsize=20,
    edge_color=edge_colors,
    edge_cmap=plt.cm.plasma,#连接配色方案
    width=4
)
#设置每个连接的透明度
edge_alphas=[(5+i)/(M+4) for i in range(M)]
for i in range (M):
    edges[i].set_alpha(edge_alphas[i])

#图例
#pc=mpl.collections.PatchCollection(edges,cmap=cmap)
#pc.set_array(edge_colors)
#plt.colorbar(pc)
ax=plt.gca()
ax.set_axis_off()
plt.show()




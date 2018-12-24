# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 18:29:17 2018

@author: Nemo
"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift
from sklearn.cluster import SpectralClustering,DBSCAN,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn import metrics




def get_data():

    final = open('Tweets.txt','r')
    
    data = [line.strip().split('\t') for line in final]
    feature = []
    cluster = []   
    for i in range(len(data)):
        temp = data[i][0]
        temp = temp.split(':')
        feature.append(temp[1][2:temp[1].index(',') - 1])
        cluster.append(int(temp[2][1:temp[2].index('}')]))
    #tokenizer: 指定分词函数; lowercase: 在分词之前将所有的文本转换成小写
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',lowercase=True)
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(feature) 
    tfidf_matrix = tfidf_matrix.toarray() 
       
    return cluster,tfidf_matrix



def kmeans(tfidf_matrix):
    num_clusters = 89 #聚类个数
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=10, 
                        init='k-means++',)
    return km_cluster.fit_predict(tfidf_matrix)
    
    
def affinityaropagation(tfidf_matrix):
    ap_cluster = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, 
        copy=True, preference=None, affinity='euclidean', verbose=False)    
    print('affinityaropagation聚类的个数：',end = "")
    print(len(set(ap_cluster.fit_predict(tfidf_matrix)))) #聚类的个数
    return ap_cluster.fit_predict(tfidf_matrix)

def meanshift(tfidf_matrix):
    ms_cluster = MeanShift(bandwidth=0.8, bin_seeding=True)
    print('meanshift聚类的个数：',end = "")
    print(len(set(ms_cluster.fit_predict(tfidf_matrix))))
    return ms_cluster.fit_predict(tfidf_matrix)

def spectralclustering(tfidf_matrix):
    num_clusters = 89 #聚类个数
    sc_cluster = SpectralClustering(n_clusters=num_clusters,
                                    assign_labels="discretize",random_state=0)
   
    return sc_cluster.fit_predict(tfidf_matrix)

def hierarchicalclustering(tfidf_matrix):
    num_clusters = 89 #聚类个数
    hc_cluster = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    return hc_cluster.fit_predict(tfidf_matrix)

   
def agglomerativeclustering(tfidf_matrix):
    num_clusters = 89
    ac_cluster = AgglomerativeClustering(n_clusters=num_clusters,linkage = 'average')
    return ac_cluster.fit_predict(tfidf_matrix)
   
def dbscan(tfidf_matrix):
    
    ds_cluster = DBSCAN(eps = 0.99, min_samples = 1 )
    print('dbscan聚类的个数：',end = "")
    print(len(set(ds_cluster.fit_predict(tfidf_matrix))))  
    return ds_cluster.fit_predict(tfidf_matrix)

def gaussianmixture(tfidf_matrix):
    num_clusters = 89
    gm_cluster = GaussianMixture(n_components = num_clusters, covariance_type='diag')
    return gm_cluster.fit(tfidf_matrix).predict(tfidf_matrix)
    
   
cluster,tfidf_matrix = get_data()


result_km = kmeans(tfidf_matrix) 
result_ap = affinityaropagation(tfidf_matrix) 
result_ms = meanshift(tfidf_matrix)
result_sc = spectralclustering(tfidf_matrix)
result_hc = hierarchicalclustering(tfidf_matrix)
result_ac = agglomerativeclustering(tfidf_matrix)
result_ds = dbscan(tfidf_matrix)
result_gm = gaussianmixture(tfidf_matrix)


print('kmeans聚类算法的成功率：',end = "")     
print(metrics.normalized_mutual_info_score(cluster, result_km ))  

print('affinityaropagation聚类算法的成功率：',end = "")    
print(metrics.normalized_mutual_info_score(cluster, result_ap)) 

print('meanshift聚类算法的成功率：',end = "")    
print(metrics.normalized_mutual_info_score(cluster, result_ms)) 

print('hierarchicalclustering聚类算法的成功率：',end = "")    
print(metrics.normalized_mutual_info_score(cluster, result_hc))

print('spectralclustering聚类算法的成功率：',end = "")    
print(metrics.normalized_mutual_info_score(cluster, result_sc))

print('agglomerativeclustering聚类算法的成功率：',end = "")    
print(metrics.normalized_mutual_info_score(cluster, result_ac))

print('dbscan聚类算法的成功率：',end = "")    
print(metrics.normalized_mutual_info_score(cluster, result_ds))

print('gaussianmixture聚类算法的成功率：',end = "")    
print(metrics.normalized_mutual_info_score(cluster, result_gm))








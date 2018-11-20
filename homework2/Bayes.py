# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:10:23 2018

@author: Nemo
"""

import os
from textblob import TextBlob 
from textblob import Word
import math


Rootdir_train = "D:\训练集" #文件的总文件名
Rootdir_test = "D:\测试集"

#获取文件名
def Return_File_Name(Rootdir):
    name = []
    
    for root,dirs,files in os.walk(Rootdir):#root遍历的文件夹
        for file in files:
            name.append(os.path.join(root,file))
    for dirname in dirs:
        Return_File_Name(dirname)
    
    return name

#获取类(文件夹)名
def file_name(file_dir):  
    list1 = []
    for root, dirs, files in os.walk(file_dir): 
        list1.append(root)
        
    del list1[0]
    for i in range(len(list1)):
        list1[i] = list1[i][7:]
    
    return list1

#文件地址
name_train = Return_File_Name(Rootdir_train)
name_test = Return_File_Name(Rootdir_test)

#文件夹名
file = file_name("D:\训练集")



#对一个字符串进行分词，规范化
def Tokenization_Stemmer(str1):
    
    zen = TextBlob(str1)    
    zen = zen.words    
    zen = zen.lemmatize()
    
    zen = list(zen)
    for i in range(len(zen)):
        w = Word(zen[i])
        zen[i] = w.lemmatize("v")
    for i in range(len(zen)):
        zen[i] = zen[i].lower()
        
    zen = sorted(zen)
     
    return zen

#停用词    
def stop_word():
    s = ""
    fopen = open("D:\停用词", 'r',errors='replace')
        
    for eachLine in fopen:
        s += eachLine
    fopen.close()
    
    stop_word = s.split()
   
    return stop_word


#获得每一个文件的字符串

def Get_str(filename):
    
    s = ""
    fopen = open(filename, 'r',errors='replace')
    
    for eachLine in fopen:
        s += eachLine
    fopen.close()
    
    return s
   
#把每一个类中出现的单词存为一个字典的key，出现的次数存为value，把所有的类的字典放到一个列表中    
def Get_dic(name):
    List = []
    
    len_name = len(name) - 1
    
    s = ""    
    for i in range(len(name)):
        if i != len_name:
            #判断此文档是否和下一个文档是否在一个类中
            if name[i][7:-7] in name[i + 1][7:-7] or name[i + 1][7:-7] in name[i][7:-7]:
                s += Get_str(name[i])
            else:
                list1 = Tokenization_Stemmer(s)
                s = ""
                a = {}
                for j in range(len(list1)):
                    if list1[j] in a.keys():
                        a[list1[j]] += 1
                    else:
                        a[list1[j]] = 1                        
                Stopword = stop_word()
                #去除停用词
                for k in range(len(Stopword)):
                    if Stopword[k] in a.keys():
                        del a[Stopword[k]]
                #把每个类的字典添加到列表中
                List.append(a)
        else:
            list1 = Tokenization_Stemmer(s)
            s = ""
            a = {}
            for j in range(len(list1)):
                if list1[j] in a.keys():
                    a[list1[j]] += 1
                else:
                    a[list1[j]] = 1
            Stopword = stop_word()
            for k in range(len(Stopword)):
                if Stopword[k] in a.keys():
                    del a[Stopword[k]]
            List.append(a)
    
    return List

List = Get_dic(name_train)

#利用朴素贝叶斯公式对测试集每个文档和训练集中每个类进行计算，返回预测得到的类的序号
def Bayes(List,name):
    s = Get_str(name)
    max_p = -float('inf')
    max_name = -1
    list1 = Tokenization_Stemmer(s)
    for i in range(len(List)):
        len_list = len(List[i])
        p = 0
        for j in range(len(list1)):
            if list1[j] in List[i].keys():
                p += math.log((List[i][list1[j]] + 1) / (len_list + 10000))#加10000进行平滑
            else:
                p += math.log(1 / (len_list + 10000 ))
        if p > max_p:
            max_p = p
            max_name = i            
    return max_name

#返回分类成功的概率           
def return_success_probability(name,file,List):

    num = 0
    sum_file = len(name)
    
    for i in range(len(name)):

        if file[Bayes(List,name[i])] in name[i]:
            num += 1        
    print(num)
    print(sum_file)
    return num / sum_file
        
print(return_success_probability(name_test,file,List))

            
        
                    
                
                
            
        





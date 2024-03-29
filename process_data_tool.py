# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
import numpy as np
import re
import sys  
import codecs  
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Installs jieba lib
# pip3 install jieba --user --default-timeout=100


stop_words_filename = "./stop_words.txt"
company_type = ["集团", "银行", "总公司", "公司", "合伙"]
jieba.analyse.set_stop_words("./stop_words.txt")

class process_data_tool:
    def __init__(self):
        # Loads stop_words file
        self.stop_words_list = list()
        f = open(stop_words_filename, "r", encoding='utf-8')
        for line in f.readlines():
            line = line.strip()
            if not len(line):
                continue
            self.stop_words_list.append(line)
        f.close()

    # Extracts data of chinese 
    def clean_data(self, data):
        # Deletes number and letter
        data = re.sub("[A-Za-z0-9]", "", data)
        # Deletes special character
        data = re.sub("[\。\，\；\《\》\？\‘\“\“\@\#\¥\%\&\（\）\——\【\】\{\}\｜\|\(\)\*\$\#\!\_\!\%\[\]\,]", "", data)
        return data

    # Compares data1 with data2
    def compare(self, data1, data2):
        data1 = self.clean_data(data1)
        data2 = self.clean_data(data2)
        if data1 == data2:
            return True
        else:
            return False
    
    def remove_stop_words(self, data):
        data = self.clean_data(data)
        data_seged = jieba.cut(data.strip(), cut_all = False) # Exact pattern
        seg_list = jieba.lcut_for_search(data.strip())
        print("====", seg_list)
        result_words = list()
        for word in data_seged:
            if word not in self.stop_words_list:
                result_words.append(word)
        return result_words
    
    # Calculates Euclidean distance
    def cal_euc_dis(self, A, B):
        return np.sqrt(np.sum(np.square(A - B)))
        # np.linalg.norm(A - B)

    # Calculates Cosine distance
    def cal_cos_dis(self, A, B):
        return 1 - np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    # Calculates similarity of two datas
    # data1 : str
    # data2 : str
    # choose : int --> 1 : Euclidean distance | 2 : Cosine distance
    def cal_similarity(self, data1, data2, choose = 2):
        TF_IDF_list1 = jieba.analyse.extract_tags(data1, topK = 10, withWeight = True)
        TF_IDF_list2 = jieba.analyse.extract_tags(data2, topK = 10, withWeight = True)
        TF_IDF_dict1 = {}
        for key, value in TF_IDF_list1:
            TF_IDF_dict1[key] = value
        TF_IDF_dict2 = {}
        for key, value in TF_IDF_list2:
            TF_IDF_dict2[key] = value
        arr1 = []
        arr2 = []
        for key in TF_IDF_dict1:
            if key in TF_IDF_dict2:
                print(key)
                arr1.append(round(float(TF_IDF_dict1[key]), 2))
                arr2.append(round(float(TF_IDF_dict2[key]), 2))
        arr1.sort(reverse = True)
        arr2.sort(reverse = True)
        print(arr1, arr2)
        if 1 == choose:
            return self.cal_euc_dis(np.array(arr1), np.array(arr2))
        elif 2 == choose:
            return self.cal_cos_dis(np.array(arr1), np.array(arr2))
        else:
            print("No choose : ", choose)
            return 0



if __name__ == '__main__':
    # print(type(jieba.analyse.extract_tags("交通银行分行西直门北大街支行", topK = 10, withWeight = True)))
    # print(jieba.analyse.extract_tags("交通银行分行西直门北大街支行", topK = 10, withWeight = True))
    tool = process_data_tool()
    # data_list = list()
    # data_list.append("中国航空工业第一集团公司科学技术委员会")
    # data_list.append("中国农业银行市分行营业部")
    # data_list.append("中国印刷总公司新华印刷厂")
    # data_list.append("一电一百广告有限公司")
    # data_list.append("上海因赛尼狄投资咨询中心普通合伙")
    # data_list.append("交通银行分行西直门北大街支行, 交通银行分行西直门支行, 交通银行分行东直门支行")
    # # data_list.append("交通银行分行西直门支行")
    # # data_list.append("交通银行分行东直门支行")
    # for data in data_list:
    #     print(data)
    #     for x, w in jieba.analyse.extract_tags(data, topK = 10, withWeight = True):
    #         print('%s %s' % (x, w))
    #     print("=============================")
        # result_words = tool.remove_stop_words(data)
        # print(result_words)

    # data1 = "交通银行分行西直门北大街支行"
    # data2 = "交通银行分行西直门支行"
    # data3 = "交通银行分行东直门支行"

    data1 = "天伦度假发展有限公司客户服务部"
    data2 = "天伦度假发展有限公司客户服务"
    data3 = "天伦度假发展有限公司第一分公司"

    # data1 = "成都农商银行市分行营业部"
    # data2 = "中国农业银行成都分行"
    # data3 = "成都农商银行分行"

    print("A1 and B = ", tool.cal_similarity(data1, data2, 1))
    print("A2 and C = ", tool.cal_similarity(data1, data3, 1))




    

        

# ['中国航空工业第一集团公司', '科学技术委员会']
# ['中国农业银行', '分行', '营业部']
# ['中国', '印刷', '总公司', '新华', '印刷厂']
# ['一电', '一百', '广告', '有限公司']
# ['上海', '因赛', '尼狄', '投资', '咨询中心', '合伙']

# TF-IDF
# =============================
# 中国航空工业第一集团公司科学技术委员会
# 中国航空工业第一集团公司 6.950338826
# 科学技术委员会 5.321290557
# =============================
# 中国农业银行市分行营业部
# 中国农业银行 3.1813229417633333
# 分行 2.59937301912
# 营业部 2.2908254066400002
# =============================
# 中国印刷总公司新华印刷厂
# 印刷厂 2.11369462836
# 总公司 1.63805012692
# 印刷 1.499089838788
# 新华 1.472797210876
# 中国 0.605464137332
# =============================
# 一电一百广告有限公司
# 一电 2.988691875725
# 一百 2.107101744575
# 广告 1.8431799335875
# 有限公司 1.2984227222325
# =============================
# 上海因赛尼狄投资咨询中心普通合伙
# 因赛 1.9924612504833332
# 尼狄 1.9924612504833332
# 咨询中心 1.7941972393333332
# 合伙 1.2362617167533334
# 上海 0.7639396102233333
# 投资 0.6483626586316666
# =============================


# Compare 交通银行分行西直门北大街支行 交通银行分行西直门支行 交通银行分行东直门支行
# =============================
# 交通银行分行西直门北大街支行 A
# 北大街 2.20206117882
# 西直门 2.19124773456
# 支行 1.72451259855
# 交通银行 1.6137590349380002
# 分行 1.559623811472
# =============================
# 交通银行分行西直门支行 B
# 西直门 2.7390596682
# 支行 2.1556407481875
# 交通银行 2.0171987936725
# 分行 1.94952976434
# =============================
# 交通银行分行东直门支行 C
# 东直门 2.6512101965
# 支行 2.1556407481875
# 交通银行 2.0171987936725
# 分行 1.94952976434
# =============================


# ============Euclidean distance=============
# A1 and B :  0.8372574275573792
# A2 and C :  0.7167984374982971
# ============Cosine distance=============
# A1 and B :  2.340292874913885e-05
# A2 and C :  1.948778866611711e-06
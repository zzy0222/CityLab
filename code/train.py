from snownlp import SnowNLP
from snownlp import sentiment
import pandas as pd
import utils
import os

dirlist = os.listdir()


def models(commands):
    words_list = utils.word_frequency_statistics()
    for comm in commands:
        if comm == 'nan':
            continue
        else:
            comm = utils.Extract_Commands(comm).extract_command()
            for w in words_list:
                if w in comm:
                    score = SnowNLP(comm)
                    # 预训练模型分类
                    if score.sentiments > 0.8:
                        with open('pos.txt', mode='a', encoding='utf-8') as p:
                            p.writelines(comm + '\n')
                    elif score.sentiments < 0.2:
                        with open('neg.txt', mode='a', encoding='utf-8') as n:
                            n.writelines(comm + '\n')
                    break


def train():
    if 'raw_data.csv' in dirlist:
        df = pd.read_csv('raw_data.csv')
    else:
        raise Exception('请先创建raw_data.csv文件')
    df.fillna('nan')
    commands = df.评论内容.dropna().tolist()
    models(commands)
    sentiment.train('neg.txt', 'pos.txt')
    sentiment.save('mysentiment.marshal')
    print('得到模型后需拷贝到snownlp的sentiment文件夹下\
        并修改__init.py__的路径来加载新权重')

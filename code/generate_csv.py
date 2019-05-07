from snownlp import SnowNLP
import pandas as pd
import utils
import os
import numpy as np
import re


class Generate_CSV():
    """docstring for Generate_CSV"""

    def __init__(self):
        super(Generate_CSV, self).__init__()
        self.dirlist = os.listdir()
        if '关键词分类.xlsx' in self.dirlist:
            self.keywords_df = pd.read_excel('关键词分类.xlsx')
        else:
            raise Exception('请先创建关键词分类.xlsx文件')
        self.date_list = ['2015/7', '2015/8', '2015/9', '2015/10',
                          '2015/11', '2015/12', '2016/1', '2016/2',
                          '2016/3', '2016/4', '2016/5', '2016/6',
                          '2016/7', '2016/8', '2016/9', '2016/10',
                          '2016/11', '2016/12', '2017/1', '2017/2',
                          '2017/3', '2017/4', '2017/5', '2017/6',
                          '2017/7', '2017/8', '2017/9', '2017/10',
                          '2017/11', '2017/12', '1月', '2月', '3月', '4月',
                          '5月', '6月', '7月', '8月', '9月',
                          '10月', '11月', '12月']

    def mark_commonds(self):
        # 这个函数将两个功能合在一起实现
        # 1.过滤掉无用评论 2.对有用评论打分
        self.dirlist = os.listdir()
        if 'raw_data.csv' in self.dirlist:
            raw_df = pd.read_csv('raw_data.csv')
            raw_df.fillna('nan')
        else:
            raise Exception('请先创建raw_data.csv文件')
        word_lists = utils.word_frequency_statistics(raw_df)
        df = pd.DataFrame(columns=['keywords', 'commands', 'date', 'mark'])
        for i, row in raw_df.iterrows():
            comm = str(row.评论内容)
            if comm == 'nan' or comm == ''or len(comm) < 2:
                continue
            else:
                comm = utils.Extract_Commands(comm).extract_command()
                if comm == '':
                    # 有可能全部是引用 没有有效评论
                    continue

            for w in word_lists:
                if w in comm:
                    mark = round(SnowNLP(comm).sentiments, 3)
                    df.loc[df.shape[0]] = [row.关键词, comm, row.评论时间, mark]
                    if df.shape[0] % 100 == 0:
                        print('已经处理%d条评论' % df.shape[0])
                    break
        print('所有评论打分完毕,正在生成sentiment_analysis.csv')
        df.to_csv('sentiment_analysis.csv', index=False, encoding='utf_8_sig')

    def commands_attribute(self):
        self.dirlist = os.listdir()
        if 'sentiment_analysis.csv' in self.dirlist:
            sen_df = pd.read_csv('sentiment_analysis.csv')
        else:
            raise Exception('请先创建sentiment_analysis.csv文件\
                或调用mark_commands方法')
        t_list = self.keywords_df.columns.tolist()
        t_list.insert(0, '日期')
        df_avg = pd.DataFrame(columns=t_list)
        df_std = pd.DataFrame(columns=t_list)
        df_sum = pd.DataFrame(columns=t_list)

        for t in self.date_list:
            df_t = sen_df[sen_df.date.str.contains(t)]
            if df_t.empty:
                continue
            list_avg = []
            list_std = []
            list_sum = []
            for c in self.keywords_df.columns:
                kwd_list = self.keywords_df[c].dropna().tolist()
                kwd_s = ''
                for i, s in enumerate(kwd_list):
                    if i == 0:
                        kwd_s += s
                    else:
                        kwd_s += '|'
                        kwd_s += s
                df_attr = df_t[df_t.keywords.str.contains(kwd_s)]
                list_avg.append(np.mean(df_attr.mark))
                list_std.append(np.std(df_attr.mark, ddof=1))
                list_sum.append(len(df_attr))
            time_list = re.findall(r'\d+\.?\d*', t)
            if '月' in t:
                t_str = '2018-' + str(time_list[0])
            else:
                t_str = str(time_list[0]) + '-' + str(time_list[1])
            list_avg.insert(0, t_str)
            list_std.insert(0, t_str)
            list_sum.insert(0, t_str)

            print('正在处理%s的数据' % t_str)
            df_avg.loc[df_avg.shape[0]] = list_avg
            df_std.loc[df_std.shape[0]] = list_std
            df_sum.loc[df_sum.shape[0]] = list_sum

        df_avg.to_csv('avg_of_mark.csv', index=False, encoding='utf_8_sig')
        df_std.to_csv('std_of_mark.csv', index=False, encoding='utf_8_sig')
        df_sum.to_csv('sum_of_commands.csv', index=False, encoding='utf_8_sig')

    def factor_weights(self):
        self.dirlist = os.listdir()
        df = pd.DataFrame(columns=['分类', '平均分权重',
                                   '标准差权重', '评论总数权重',
                                   '归一化的总权重'])
        if 'avg_of_mark.csv' in self.dirlist and \
            'std_of_mark.csv' in self.dirlist and\
                'sum_of_commands.csv' in self.dirlist:
            df_avg = pd.read_csv('avg_of_mark.csv')
            df_avg = df_avg.dropna().reset_index(drop=True)
            df_std = pd.read_csv('std_of_mark.csv')
            df_std = df_avg.dropna().reset_index(drop=True)
            df_sum = pd.read_csv('sum_of_commands.csv')
            df_sum = df_avg.dropna().reset_index(drop=True)
        else:
            raise Exception('请先创建avg_of_mark.csv, std_of_mark.csv, \
                sum_of_commands.csv文件 或调用commands_attribute方法')

        print('正在计算平均分权重…')
        df_avg_indexs = df_avg.columns[1:].tolist()
        df_avg_positive = df_avg_indexs
        df_avg_negative = []
        df_avg_date = df_avg['日期']
        df_avg_index = df_avg[df_avg_indexs]

        df_avg_en = utils.EntropyMethod(df_avg_index, df_avg_negative,
                                        df_avg_positive, df_avg_date)
        avg_series = df_avg_en.calc_Weight()
        df.平均分权重 = avg_series

        print('正在计算标准差权重…')
        std_wl = []
        for c in df_std.columns[1:]:
            std_wl.append(np.mean(df_std[c]))
        std_ws = pd.Series(std_wl)
        std_ws = std_ws / np.sum(std_ws)
        df.标准差权重 = std_ws.tolist()

        print('正在评论总数权重…')
        sum_wl = []
        for c in df_sum.columns[1:]:
            sum_wl.append(np.mean(df_sum[c]))
        sum_ws = pd.Series(sum_wl)
        sum_ws = std_ws / np.sum(sum_ws)
        df.评论总数权重 = sum_ws.tolist()

        print('正在计算归一化的总权重…')
        df.分类 = df_avg.columns[1:]
        ws = df.平均分权重 * df.标准差权重 * df.评论总数权重
        ws = ws / np.sum(ws)
        df.归一化的总权重 = ws

        df.to_csv('factor_weights.csv', index=False, encoding='utf_8_sig')

    def total_grade(self):
        self.dirlist = os.listdir()
        if 'avg_of_mark.csv' in self.dirlist and \
                'factor_weights.csv' in self.dirlist:
            df_score = pd.read_csv('avg_of_mark.csv')
            df_weight = pd.read_csv('factor_weights.csv')
        else:
            raise Exception('请先创建avg_of_mark.csv, factor_weights.csv\
                或调用factor_weights方法')
        t_list = df_score.columns.tolist()
        t_list.append('满意度综合指数')
        df = pd.DataFrame(columns=t_list)
        years = ['2015', '2016', '2017', '2018']

        print('正在计算4年的满意度综合指数…')
        for y in years:
            df_year = df_score[df_score.日期.str.contains(y)]
            score_array = np.array(np.mean(df_year))
            weight_array = np.array(df_weight.归一化的总权重)
            col_list = np.multiply(score_array, weight_array).tolist()
            col_list.append(np.sum(col_list))
            col_list.insert(0, y)
            df.loc[df.shape[0]] = col_list

        df.to_csv('total_grade.csv', index=False, encoding='utf_8_sig')

    def gary_relational_analysis(self):
        self.dirlist = os.listdir()
        if 'total_grade.csv' in self.dirlist and\
                'annual.csv' in self.dirlist:
            df_satisfy = pd.read_csv('total_grade.csv')
            df_annual = pd.read_csv('annual.csv')
        else:
            raise Exception('请先创建total_grade.csv, \
                annual.csv 或调用total_grade方法')

        list_df = df_annual.columns.tolist()
        list_df.insert(1, df_satisfy.columns[-1])
        df = pd.DataFrame(columns=list_df)

        print('正在进行灰色关联度分析…')
        df[df_satisfy.columns[-1]] = df_satisfy.iloc[:, -1]
        for col in df_annual.columns:
            df[col] = df_annual[col]
        gra = utils.Gray_Relational_Analysis(df_annual)
        gra.gray_analysis()
        gra.RT.to_csv('gary_relational_analysis.csv',
                      index=False, encoding='utf_8_sig')

        print('正在生成灰色关联度分析热力图…')
        gra.show_gra_heatmap()

    def process_all(self):
        self.mark_commonds()
        self.commands_attribute()
        self.factor_weights()
        self.total_grade()
        self.gary_relational_analysis()

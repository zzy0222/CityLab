import re
import pandas as pd
import numpy as np


class Extract_Commands():
    # 提取微博评论的内容
    def __init__(self, content):
        super(Extract_Commands, self).__init__()
        self.content = content
        if content == '':
            raise Exception('评论内容为空')

    def clear_quotation(self):
        if self.content[0] == '回' or self.content[0] == '@':
            self.content = re.sub('[@回](.*)[: ]', '', self.content)
        else:
            self.content = re.sub('[@回](.*)', '', self.content)

    def clear_special_character(self):
        if '/' in self.content:
            self.content = re.sub('/', '', self.content)
        if '#' in self.content:
            self.content = re.sub('#(.*)#', '', self.content)

    def extract_command(self):
        self.clear_special_character()
        if self.content != '':
            self.clear_quotation()
        return self.content


def word_frequency_statistics(raw_data):
    categorys = raw_data.关键词.unique()
    words_list = []
    # 分类统计词频
    for c in categorys:
        df_new = raw_data[(raw_data.关键词 == c)]
        df_new.dropna(inplace=True)
        cuts = df_new.分词后评论内容

        counts = {}
        for words in cuts:
            if isinstance(words, str):
                word_list = words.split(' ')
            for w in word_list:
                if w == '' or w == '回复' or len(w) == 1:
                    continue
                else:
                    counts[w] = counts.get(w, 0) + 1
        items = list(counts.items())
        items.sort(key=lambda x: x[1], reverse=True)

        print('正在生成 %s 的关键词库' % c)
        for i in range(20):
            w, count = items[i]
            words_list.append(w)
    words_list = list(set(words_list))  # 去重

    # 使用中文停用词表对关键词进行筛查
    with open('中文停用词1208.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line in words_list:
                del words_list[words_list.index(line)]
    return words_list


class EntropyMethod():
    def __init__(self, index, positive, negative, row_name):
        if len(index) != len(row_name):
            raise Exception('数据指标行数与行名称数不符')
        if sorted(index.columns) != sorted(positive + negative):
            raise Exception('正项指标加负向指标不等于数据指标的条目数')

        self.index = index.copy().astype('float64')
        self.positive = positive
        self.negative = negative
        self.row_name = row_name

    def uniform(self):
        uniform_mat = self.index.copy()
        min_index = {column: min(uniform_mat[column])
                     for column in uniform_mat.columns}
        max_index = {column: max(uniform_mat[column])
                     for column in uniform_mat.columns}
        for i in range(len(uniform_mat)):
            for column in uniform_mat.columns:
                if column in self.negative:
                    uniform_mat[column][i] = (
                        uniform_mat[column][i] - min_index[column]) \
                        / (max_index[column] - min_index[column])
                else:
                    uniform_mat[column][i] = (
                        max_index[column] - uniform_mat[column][i]) \
                        / (max_index[column] - min_index[column])

        self.uniform_mat = uniform_mat
        return self.uniform_mat

    def calc_probability(self):
        try:
            p_mat = self.uniform_mat.copy()
        except AttributeError:
            raise Exception('你还没进行归一化处理，请先调用uniform方法')
        for column in p_mat.columns:
            sigma_x_1_n_j = sum(p_mat[column])
            p_mat[column] = p_mat[column].apply(
                lambda x_i_j: x_i_j / sigma_x_1_n_j if x_i_j
                / sigma_x_1_n_j != 0 else 1e-6)

        self.p_mat = p_mat
        return p_mat

    def calc_emtropy(self):
        try:
            self.p_mat.head(0)
        except AttributeError:
            raise Exception('你还没计算比重，请先调用calc_probability方法')

        import numpy as np
        e_j = -(1 / np.log(len(self.p_mat) + 1)) * np.array([sum([pij * np.log(
            pij) for pij in self.p_mat[column]])
            for column in self.p_mat.columns])
        ejs = pd.Series(e_j, index=self.p_mat.columns, name='指标的熵值')

        self.emtropy_series = ejs
        return self.emtropy_series

    def calc_emtropy_redundancy(self):
        try:
            self.d_series = 1 - self.emtropy_series
            self.d_series.name = '信息熵冗余度'
        except AttributeError:
            raise Exception('你还没计算信息熵，请先调用calc_emtropy方法')

        return self.d_series

    def calc_Weight(self):
        self.uniform()
        self.calc_probability()
        self.calc_emtropy()
        self.calc_emtropy_redundancy()
        self.Weight = self.d_series / sum(self.d_series)
        self.Weight.name = '权值'
        return self.Weight

    def calc_score(self):
        self.calc_Weight()

        import numpy as np
        self.score = pd.Series(
            [np.dot(np.array(self.index[row:row + 1])[0],
                    np.array(self.Weight))
             for row in range(len(self.index))],
            index=self.row_name, name='得分'
        ).sort_values(ascending=False)
        return self.score


class Gray_Relational_Analysis():
    def __init__(self, df, m=0):
        super(Gray_Relational_Analysis, self).__init__()
        self.df = df
        self.m = m

    def gray_analysis(self):
        gray = (self.df - self.df.min()) \
            / (self.df.max() - self.df.min())

        std = gray.iloc[:, self.m]  # 为标准要素
        ce = gray.iloc[:, 0:]  # 为比较要素
        n = ce.shape[0]
        m = ce.shape[1]  # 计算行列

        # 与标准要素比较，相减
        a = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                a[i, j] = abs(ce.iloc[j, i] - std[j])

        # 取出矩阵中最大值与最小值
        c = np.amax(a)
        d = np.amin(a)

        # 计算值
        result = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                result[i, j] = (d + 0.5 * c) / (a[i, j] + 0.5 * c)

        # 求均值，得到灰色关联值
        result2 = np.zeros(m)
        for i in range(m):
            result2[i] = np.mean(result[i, :])
        t_list = result2.tolist()
        del t_list[0]  # 删除年份
        # 相关度向量
        RT = pd.DataFrame(t_list)
        # 用来画图的RT_plt
        RT = RT.rename(columns={0: "相关程度百分比"})
        RT['客观指标'] = self.df.columns[1:]
        RT_plt = RT.pivot_table(index='客观指标', values='相关程度百分比')
        RT_plt = RT_plt.sort_values(by='相关程度百分比', ascending=False)

        self.RT = RT
        self.RT_plt = RT_plt

    def show_gra_heatmap(self):
        self.gray_analysis()
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(font='simhei')
        plt.title('灰色关联度分析热力图', y=1.05, size=15)
        plt.xlabel('客观指标')
        plt.ylabel('相关程度百分比')
        sns.heatmap(self.RT_plt, linewidths=0.1, vmax=1.0,
                    fmt='.0%', cmap="YlGnBu", linecolor='white', annot=True)
        plt.tight_layout()
        plt.savefig("灰色关联度分析热力图.jpg",)
        plt.show()

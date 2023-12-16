[English](#description) | [中文](#描述)

# CityLab
## 描述
2019年北京高校数学建模校际联赛B题的模型代码：NLP情感分析+改进熵值法+改进灰色关联度分析<br>
原来代码因为误删.git 所以又重新传了一遍<br>
由于论文还没有评奖，所以暂时先不放论文啦~<br>
大致思路：<br>
1. nlp情感分析对有用评论打分，区间[0,1]<br>
2. 熵值法建模，一共八个影响因素（关键词分类.xlsx），每个影响因素用3个因子衡量（详见factor_weights方法）<br>
3. 灰色关联度分析，找客观指标对主观感受（满意度幸福指数）的影响效果<br>

## 更新
2019-5-14 评审结果是一等奖！

## 代码使用说明
一级目录下是一些原始数据，主要讲一下三个文件夹。

### code
1. demo文件夹里包含代码运行的全部结果，大家可以拿来对比和测试类中的方法<br><br>
2. 4个.py的文件<br><br>
   generate_csv.py：用来生成demo中的全部csv文件 大家查看代码就知道*执行后面的方法必须要有前面生成的.csv文件*<br><br>
   train.py：用来train情感打分模型。由于建模的时间关系，当时只train了一次。大家可以改代码多train几次，效果应该会更好<br><br>
   utils.py：工具类。里面实现了熵值法和灰色关联分析法，还有提取评论的函数。（因为本人的正则实在不行，欢迎各位dalao fork修改）<br><br>
   main.py：直接python main.py就可以跑了。里面调用的是process_all方法，要是想看中间过程，按照generate_csv.py里面的方法自行修改即可<br><br>
3. 3个数据表<br><br>
   annual.csv：2015 - 2018年部分北京统计年鉴的内容摘录<br><br>
   raw_data.csv：原始的B题数据<br><br>
   关键词分类.xlsx：对原始B题数据中“关键词”一项进行*人工分类*以后的结果<br><br>
4. 中文停用词1208.txt：1208个中文停用词，用来筛选分词评论的

### data_csv
里面的五个文件夹代表在模型的不同过程生成的数据表。这里面不带有"_v2"的csv和demo中的一致。<br>
带有"_v2"的csv文件程序无法生成，是将情感打分区间换算到[-5,5];原始的情感打分区间是[0,1]

### models
neg.txt和pos.txt是用nlp原始的情感分析模型（用的是电商评论语料库，所以要重新train）对有用评论做的一个情感倾向的划分（只有正负）<br>
my_sentiment.marshal.3是train后的模型，需要拷贝到snownlp下面的sentiment文件夹下，并修改__init.py__的模型加载路径<br>

## 致谢
最后十分感谢北京师范大学刘虓，宗一博两位同伴在建模过程中的对我的支持和帮助，这是我们共同的荣誉。<br><br><br>


## Description
Model code for the 2019 Beijing College Mathematical Modeling Intercollegiate Competition Question B: NLP Sentiment Analysis + Improved Entropy Method + Improved Grey Relational Analysis<br>
The original code was re-uploaded due to accidental deletion of the .git file<br>
Since the paper hasn't been awarded yet, it won't be released for now<br>
General approach:<br>
NLP sentiment analysis to score useful comments, range [0,1]<br>
Entropy method modeling with eight influencing factors (see keywords_classification.xlsx), each factor measured by three elements (see factor_weights method)<br>
Grey relational analysis to identify the impact of objective indicators on subjective feelings (satisfaction and happiness index)<br>

## Updates
2019-5-14 Received first prize in the review!

## Code Usage Instructions
The root directory contains some raw data, but the main focus is on three folders.

### code
The demo folder contains all the results of the code execution, which you can use for comparison and testing<br><br>
4 .py files<br><br>
generate_csv.py: Generates all the csv files in the demo, showing that subsequent methods require the .csv files generated previously<br><br>
train.py: Used for training the sentiment scoring model. Due to time constraints during modeling, it was only trained once. You can modify the code for more training sessions for potentially better results<br><br>
utils.py: A utility class. Implements the entropy method and grey relational analysis method, as well as functions for extracting comments. (Since my regular expressions are not strong, I welcome any experts to fork and modify)<br><br>
main.py: Run with python main.py. It calls the process_all method. If you want to see the intermediate process, modify it according to the methods in generate_csv.py<br><br>
3 data tables<br><br>
annual.csv: Excerpts from the Beijing Statistical Yearbook from 2015 - 2018<br><br>
raw_data.csv: Original data for Question B<br><br>
关键词分类.xlsx: Results of manual classification of the “关键词” in the original Question B data<br><br>
中文停用词1208.txt: 1208 Chinese stop words for filtering segmented comments

### data_csv
The five folders here represent data tables generated at different stages of the model. The csv files without "_v2" are consistent with those in the demo.<br>
The "_v2" csv files, which the program cannot generate, have sentiment scores converted to the range [-5,5]; the original sentiment scoring range is [0,1]

### models
neg.txt and pos.txt are the initial sentiment analysis models (using e-commerce review corpora, hence the need for retraining) for categorizing useful comments (just positive and negative)<br>
my_sentiment.marshal.3 is the trained model, which needs to be copied to the sentiment folder in snownlp and modify the model loading path in __init.py__<br>

## Acknowledge
Finally, I would like to express my heartfelt thanks to Liu Xiao and Zong Yibo from Beijing Normal University for their support and help during the modeling process. This is an honor we share together.

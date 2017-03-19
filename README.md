add a line in master 
# weibo_emotion
情感分析项目

描述： 数据集总共有4个class, 每个文件里的都属于同一个class，不同数字对应不同类别的感情：

0：喜悦（比如数字0开头的文件0_simplifyweibo_train为对应喜悦情绪的数据）

1：愤怒

2：厌恶

3：低落

针对于每一个文件，每一行都是一条微博， 这些都是已分词好的，而且对于每个单词，已给出POS tag。 

目标：针对于测试数据中的每一条微博，预测情感的类型（四类中的某一类）

评测：本次作业的评测标准是准确率，也就是针对于每一条微博先预测出它的label, 然后把预测的label和实际的label做比较。比如10条微博当中，准确预测出6条微博的label, 那准确率为0.6


附加的数据文件ADDITIONAL DATA FILES
本次作业的训练数据（4个文件）以及项目描述doc文档，都以压缩包形式放在附加文件中。包括：

project2.docx

0_simplifyweibo_train.txt

1_simplifyweibo_train.txt

2_simplifyweibo_train.txt

3_simplifyweibo_train.txt

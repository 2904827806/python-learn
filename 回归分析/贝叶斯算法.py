#贝叶斯算法

"""
正向概率：假设袋子里面有n个白球，m个黑球，伸手去摸，摸出黑球的概率是多大

逆向概率：如果我们事先并不知道袋子里面黑球白球的比例，而是闭着眼睛摸出一个球，观察这些取出
的球的颜色之后，那么我们可以就此对袋子里面的黑白球比例作出什么样的推断

公式：P(A|B) = (P(B|A) * P(A)) / P(B)


"""

#拼写纠正案例
#问题是我们看到用户输入了一个不存在字典中的单词，我们需要去
#猜测：他到底想输入的单词书什么？

#p(我们猜测他输入的单词|他实际输入的单词)
"""
假设用户输入的单词记为D（D代表Data，就是观察数据）
猜测1：p（h1|D），猜测2：p（h2|D），猜测3：p（h3|D）
统一为：p(h|D) 

p(h|D) = p（h)*p(D|h)/p(D)   #p(h)是特定猜测的先验概率 ,p(D)每次都一样，可以省略
对于给定观察数据，猜测好坏取决于猜测本身独立的可能性大小（先验概率，Prior）
和这个猜测生成我们观察数据可能性大小

"""

#最大似然：最符合观测数据的（p(h|D) 最大的）最有优势

#奥卡姆剃刀：p(h)较大的模型有较大的优势（越高级的多项式越不常见）

#贝叶斯拼写检查器 根据步距获取p(D|h)
import re,collections
def words(text):
    #把材料中的单词全转换为小写，并去除单词中间的特殊符号
    return re.findall('[a-z]+',text.lower())  #返回查找结果的列表.lower()小写
def train(features): #统计每个词出现的次数
    #接受一个特征列表（例如单词列表），然后返回一个defaultdict，其中每个单词作为键，其出现的次数作为值。
    # 注意这里lambda : 1意味着当字典中不存在某个键时，其默认值为1。
    model = collections.defaultdict(lambda:1) #最少出现1次
    for f in features:
        model[f] += 1
    return model

#输入经验数据库
NWORDS = train(words(open(r"C:\Users\29048\Desktop\big.txt",encoding='utf-8').read())) #统计数据中单词出现的次数
#print(NWORDS)
alphabet = 'abcdefghijklmnopqrstuvwxyz' #设置查找特征

#编辑距离
def editsl(word):
    #函数返回给定单词w编辑距离为1的集合。
    n = len(word)
    #增删改查
    b = set([word[0:i]+word[i+1:]for i in range(n)] +
               [word[0:i] + word[i+1]+word[i]+word[i+2:] for i in range(n-1)]+
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet]+
               [word[0:i] + c + word[i + 1:] for i in range(n+1) for c in alphabet])
    #print(b)
    return b  #返回获取的词
def known_edits2(word):#编辑距离为2 的单词
    return set(e2 for e1 in editsl(word) for e2 in editsl(e1) if e2 in NWORDS)

def known(words):#将那些正确的词作为候选词
    #返回给定单词列表中所有存在于NWORDS中的单词。
    return set(w for w in words if w in NWORDS)

#如果know（set）非空
def correct(word): #检查器函数，先判断是不是正确的拼写形式，如果不是则选出编辑距离为1的单词……
    #函数是拼写校正的核心。它首先尝试查找与给定单词完全匹配的单词。
    # 如果没有找到，它会尝试查找通过一次编辑得到的单词。如果还没有找到，
    # 它会尝试查找通过两次编辑得到的单词。最后，如果所有方法都失败，
    # 它会返回原始的单词。
    # 在所有步骤中，它都优先选择NWORDS中出现次数最多的单词作为校正结果。
    candidates = known([word]) or known(editsl(word)) or known(known_edits2(word)) or [word]
    return max(candidates,key=lambda w: NWORDS[w])  #根据NWORDS中的概率获取candidates中出现概率最大的词

a = correct('appla')



#文本分析
#关键词提取
#停用词
   #1.语料中大量出现
   #2.没什么用

#词频 = 某个词在文章中出现的次数 / 文章总字数 = TF

#逆文档频率 = log2（语料库的文档总数/（包含该词的文档数+1）） = IDF

#关键词提取 tf-idf = 词频*逆文档频率 =TF*IDF

#相似度 :余弦相似度



# 导入库
import re  # 正则表达式库
import collections  # 词频统计库
import numpy as np  # numpy库
import jieba  # 结巴分词
import wordcloud  # 词云展示库
from PIL import Image  # 图像处理库
import matplotlib.pyplot as plt  # 图像展示库
import jieba.analyse  # 导入关键字提取库
import pandas as pd

# 读取文本文件
with open(r"C:\Users\29048\Desktop\data-analysis-material-master\数据挖掘\文本分析\article1.txt", encoding='gbk') as fn:
    string_data = fn.read()  # 使用read方法读取整段文本

# 文本预处理
pattern = re.compile(u'\t|\n|\.|-|一|:|;|\)|\(|\?|"')  # 建立正则表达式匹配模式
string_data = re.sub(pattern, '', string_data)  # 将符合模式的字符串替换掉


# 文本分词

"""
# 文本分词 jieba分词器

content = string_data.content.values.tolist() 
content_S = []
for line in content:
    content_segmengt = jieba.lcut(line)
    if len(content_segmengt) > 1 and content_segmengt !='\r\n':
        content_S.append(content_segmengt)"""

seg_list_exact = jieba.cut(string_data, cut_all=False)  # 精确模式分词[默认模式]

remove_words = ['的', '，', '和', '是', '随着', '对于', ' ', '对', '等', '能',
                '都', '。', '、', '中', '与', '在', '其', '了', '可以',
                '进行', '有', '更', '需要', '提供', '多', '能力', '通过',
                '会', '不同', '一个', '这个', '我们', '将', '并', '同时',
                '看', '如果', '但', '到', '非常', '—', '如何', '包括', '这']  # 自定义停用词
object_list = [i for i in seg_list_exact if i not in remove_words] # 将不在停用词列表中的词添加到列表中
# 词频统计
word_counts = collections.Counter(object_list)  # 对分词做词频统计

word_counts_top5 = word_counts.most_common(5)  # 获取前5个频率最高的词
for w, c in word_counts_top5:  # 分别读出每条词和出现从次数
    print(w, c)  # 打印输出
# 词频展示
mask = np.array(Image.open(r"C:\Users\29048\Desktop\2582.jpg"))  # 定义词频背景
wc = wordcloud.WordCloud(
    font_path=r"C:\Users\29048\Desktop\fzztdqdb_downcc\fangzhengzitiku\FZBSJW.TTF",  # 设置字体格式，不设置将无法显示中文
    mask=mask,  # 设置背景图,
    background_color='white',
    max_words=200,  # 设置最大显示的词数
    max_font_size=100  # 设置字体最大值
)
word_frequence = {x[0]: x[1] for x in word_counts.items()}
wo = wc.fit_words(word_frequence)
wc.generate_from_frequencies(word_counts)  # 从字典生成词云
image_colors = wordcloud.ImageColorGenerator(mask)  # 从背景图建立颜色方案
wc.recolor(color_func=image_colors)  # 将词云颜色设置为背景图方案
plt.figure(figsize=(12,8))
plt.imshow(wc)  # 显示词云
plt.imshow(wo)  # 显示词云
plt.axis('off')  # 关闭坐标轴
plt.show()


# 读取文本数据
with open(r"C:\Users\29048\Desktop\data-analysis-material-master\数据挖掘\文本分析\article1.txt", encoding='gbk') as fn:
    string_data = fn.read()  # 使用read方法读取整段文本
# 关键字提取
#使用jieba这个Python库来进行中文分词和关键词提取
# jieba.analyse.extract_tags函数来从string_data中提取关键词
#topK=5: 表示你想要提取的关键词数量
#withWeight=True: 表示返回的结果中，每个关键词都会附带其权重。权重通常表示该词在文本中的重要性或频率。
tags_pairs = jieba.analyse.extract_tags(string_data, topK=5, withWeight=True,
                                        allowPOS=['ns', 'n', 'vn', 'v', 'nr'], withFlag=True)  # 提取指定词性
print(tags_pairs)
tags_list = [(i[0].word, i[0].flag, i[1]) for i in tags_pairs]
tags_pd = pd.DataFrame(tags_list, columns=['word', 'flag', 'weight'])  # 创建数据框
print(tags_pd)

#LDA :主题模型

'''
# 格式要求，分词好整个语料

# 从gensim库中导入corpora（用于处理语料库）、models（包含各种模型类）、
# similarities（用于计算文档之间的相似性）模块
from gensim import corpora, models, similarities
# 导入gensim库本身，这里导入并没有在后续代码中直接使用，所以可能是不必要的
import gensim

# 做映射，相当于词袋
# 使用corpora模块的Dictionary类来创建一个字典对象，该对象将文本中的词汇映射为唯一的整数ID
# object_list应该是包含文本数据的列表，每个元素是一个句子（通常需要先进行分词处理） 上述表示列表
object_list = pd.DataFrame(object_list)
dictionary = corpora.Dictionary(object_list) #字典

# 使用字典对象将object_list中的每个句子转换为词袋表示（bag-of-words），即每个词及其在该句子中出现的次数
# doc2bow函数返回一个元组的列表，其中每个元组包含词的ID和其在句子中的频率
corpus = [dictionary.doc2bow(sentence) for sentence in object_list]

# 创建LDA模型
# 使用了gensim.models.ldamodel.LdaModel类（注意：在新版本的gensim中，通常直接使用gensim.models.LdaModel）
# 传入语料库corpus、词汇到ID的映射dictionary以及主题数量num_topic（这里应该是num_topics，是一个常见的错误）
# 注意：这里使用了错误的参数名num_topic，正确的应该是num_topics
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

# 一号分类结果
# 尝试打印LDA模型中的第一个主题及其前5个最重要的词汇
# 注意：这里使用了错误的函数名print_topic和错误的参数位置，正确的应该是print_topics和参数应该在函数内部
# 并且，num_words参数用于指定每个主题要打印的词汇数量
# 这里应该是print(lda.print_topics(num_topics=1, num_words=5))，来打印第一个主题的前5个词汇
print(lda.print_topics(num_topics=1, num_words=5))

#遍历输出数据
for topic in lda.print_topics(num_topics=20, num_words=5):
    print(topic)

'''

#对数据进行分类任务
# 创建一个pandas DataFrame，包含两列：'contens_clean'（内容为object_list）和'label'（内容为string_data['category']的值）
de_F = pd.DataFrame({'contens_clean': object_list, 'label': string_data['category']})

# 显示DataFrame的最后几行，通常用于快速检查数据
de_F.tail()

# 获取'label'列中所有唯一值，并存储在变量a中
a = de_F.label.unique()

# 创建一个标签映射字典，将原始标签映射为数字（例如，'汽车'映射为1，'财经'映射为2等）
label_mapping = {'汽车': 1, '财经': 2, '科技': 3, '健康': 4, '体育': 5, '教育': 6, '文化': 7, '军事': 8, '娱乐': 9, }

# 使用map函数将DataFrame中的'label'列替换为对应的数字标签
de_F['label'] = de_F['label'].map(label_mapping)

# 从sklearn.model_selection中导入train_test_split函数，用于将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split

# 使用train_test_split函数拆分'contens_clean'列（特征）和'label'列（目标变量）为训练集和测试集
# 但这里有一个错误，应该接收四个返回值，而不是直接赋值给x_tr, y_tr, x_ts, y_ts
# 正确的赋值应该是 x_tr, x_ts, y_tr, y_ts = ...
x_tr, y_tr, x_ts, y_ts = train_test_split(de_F['contens_clean'].values, de_F['label'].values)

# 初始化一个空列表words1，意图不明，但看起来像是想要对训练数据进行某种处理
words1 = []

# 这是一个错误的循环，x_tr是一个NumPy数组，不能直接用于range()函数
# 此外，x_tr[line_index]已经是字符串，不需要''.join(x_tr[line_index])
for line_index in range(x_tr):
    try:
        words1.append(''.join(x_tr[line_index]))
    except:
        print(line_index, 'word_index')

# 从sklearn.feature_extraction.text中导入CountVectorizer类，用于文本特征提取
from sklearn.feature_extraction.text import CountVectorizer

# 定义一个示例文本列表texts，用于演示CountVectorizer的使用
texts = ['dog cat fish', 'dog cat cat', 'fish bird', 'bird']

# 创建一个CountVectorizer对象cv
cv = CountVectorizer()

# 使用fit方法拟合texts数据，并返回一个转换器对象cv_fit（尽管这里cv和cv_fit实际上是同一个对象）
cv_fit = cv.fit(texts)

# 打印出CountVectorizer从texts中提取的所有特征名称（即，唯一的单词）
print(cv.get_feature_names_out())

# 将texts转换为特征矩阵（这里texts很小，所以使用toarray()返回密集数组）
# 但通常对于大数据集，我们会使用稀疏表示
print(cv_fit.toarray())

# 打印特征矩阵中每个特征（即单词）在所有文档中出现的次数总和
print(cv_fit.toarray().sum(axis=0))

# 导入CountVectorizer类，用于将文本数据转换为特征向量
from sklearn.feature_extraction.text import CountVectorizer

# 创建一个CountVectorizer对象，配置如下：
# - 分析器(analyzer)设置为'word'，表示以单词为单位进行特征提取
# - 最大特征数(max_features)设置为4000，表示最多保留4000个最常见的单词作为特征
# - lowercase设置为False，表示不将文本转换为小写
vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)

# 使用words1列表中的文本数据来训练CountVectorizer，即确定哪些单词是特征
vec.fit(words1)

# 导入MultinomialNB类，这是一个朴素贝叶斯分类器，用于分类文本数据
from sklearn.naive_bayes import MultinomialNB

# 创建一个MultinomialNB对象
classifier = MultinomialNB()

# 使用CountVectorizer将words1中的文本转换为特征向量，并用这些特征向量和对应的y_tr标签来训练分类器
classifier.fit(vec.transform(words1), y_tr)

# 这里存在一个逻辑错误，words1列表在后面的循环中再次被使用来追加x_ts中的数据
# 但这通常是不正确的，因为words1应该是用于训练的文本数据
# 下面这段循环试图将测试集x_ts中的数据追加到words1中，这是不必要的且错误的
test_words = []
for line_index in range(x_ts):
    try:
        # 注意：这里尝试将x_ts中的数据追加到words1，而不是test_words
        # 并且x_ts[line_index]已经是字符串，不需要''.join(x_ts[line_index])
        words1.append(''.join(x_ts[line_index]))
    except:
        print(line_index, 'word_index')

    # 接下来的代码段与之前的代码段重复，并且也存在同样的问题
# 这可能是一个错误，或者是不必要的复制粘贴

# 重复创建了一个CountVectorizer对象，这通常是不必要的
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)

# 再次使用words1（现在包含了测试集数据）来训练CountVectorizer
# 这会改变特征集，因为训练集和测试集的特征应该是一致的
vectorizer.fit(words1)

# 重复创建了一个MultinomialNB对象，这也是不必要的
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

# 使用修改后的words1（包含测试集数据）和相同的y_tr标签来训练分类器
# 这会导致分类器在训练时看到测试集的数据，这是不正确的
classifier.fit(vectorizer.transform(words1), y_tr)

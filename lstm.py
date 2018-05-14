# scikit-learn 須升級至 0.19
# pip install -U scikit-learn
# 在 python 執行 nltk.download(), 下載 data
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections #用来统计词频
import nltk #用来分词
import numpy as np
from keras.models import load_model
nltk.download('punkt')
## 探索数据分析(EDA)
# 了解数据中有多少个不同的单词，每句话由多少单词组成
maxlen = 0 #句子最大长度
word_freqs = collections.Counter() #词频
num_recs = 0 #样本数
with open('train_data.txt', 'r+', encoding='UTF-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())#查查查
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ', maxlen) #每句话最多包含的单词量
print('nb_words ', len(word_freqs)) #一共多少个不同的单词 包括标点符号
"""根据不同单词的个数 (nb_words)，我们可以把词汇表的大小设为一个定值，
并且对于不在词汇表里的单词，把它们用伪单词 UNK 代替。
根据句子的最大长度 (max_lens)，我们可以统一句子的长度，把短句用 0 填充"""
## 准备数据
"""我们把 VOCABULARY_SIZE 设为 2002。包含训练数据中按词频从大到小排序后的前 2000 个单词，
，外加一个伪单词 UNK 和填充单词 0。最大句子长度 MAX_SENTENCE_LENGTH 设为40。"""
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40
"""建立两个 lookup tables，分别是 word2index 和 index2word，用于单词和数字转换。"""
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v: k for k, v in word2index.items()}
"""根据 lookup table 把句子转换成数字序列，并把长度统一到 MAX_SENTENCE_LENGTH，不够的填 0 ，多出的截掉。 """
X = np.empty(num_recs, dtype=list)
y = np.zeros(num_recs)
i = 0
# 读取训练资料，将每一单字以dictionary存储
with open('train_data.txt', 'r+', encoding='UTF-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1

# 字句长度不足补空白
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
# 划分数据，80%作为训练数据，20%作为测试数据
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
# 模型构建
"""模型构建：损失函数为binary_crossentropy；优化方法用adam 
EMBEDDING_SIZE , HIDDEN_LAYER_SIZE，BATCH_SIZENUM_EPOCHS 凭经验定"""
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64

model = Sequential()
# 加入嵌入层
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
# 加入LSTM层
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
# binary_crossentropy:二分法
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# 模型訓練
"""用10个epochs和batch_size取32来训练
在每个epoch，用测试集当做验证集"""
BATCH_SIZE = 32
NUM_EPOCHS = 10
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest))

# 预测
"""用已经训练好的LSTM预测已经划分好的测试集的数据 
随机选取6个句子进行预测"""
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('预测', '真实', '句子'))
for i in range(6):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1, MAX_SENTENCE_LENGTH)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))

# 模型存檔
model.save('Sentiment1.h5')  # creates a HDF5 file 'model.h5'

##### 自己輸入測試
INPUT_SENTENCES = ['I love it.', 'It is so boring.', 'I love it althougn it is so boring.']
XX = np.empty(len(INPUT_SENTENCES), dtype=list)
# 轉換文字為數值
i = 0
for sentence in INPUT_SENTENCES:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i += 1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
# 預測，並將結果四捨五入，轉換為 0 或 1
labels = [int(round(x[0])) for x in model.predict(XX)]
label2word = {1: '正面', 0: '負面'}
# 顯示結果
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))
# Imdb
Imdb prediction
## Data Source
Data Setsets: [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)

Download link: [http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

## Step
1. 读取 IMDB 数据集
2. 建立 token 字典
3. 使用 token 将"影评文字"转换为"数字列表"
4. 截长补短让所有"数字列表"长度都是100
5. Embedding 层将"数字列表"转换为"向量列表"
6. 将"向量列表"送入深度学习模型进行训练

## Models
MLP 和 CNN 都只能按照当前的状态进行识别，如果要处理时间序列的问题，就必须使用 RNN 与 LSTM 模型
1. MLP
2. CNN
3. RNN (长期依赖问题)
```
#RNN Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.embedding import Embedding
from keras.layers.recurrent import SimpleRNN
model = Sequential()
model.add(Embedding(output_dim =32, input_dim=3800, input_length=380))
model.add(Dropout(0.35))
model.add(SimpleRNN(units=16))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(units=1,activation='sigmoid'))
scores=model.evaluate(x_test, y_test, verbose=1)
scores[1]
```
4. LSTM
 ```
#LSTM Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers.embedding import Embedding
from keras.layers.recurrent import LSTM
model = Sequential()
model.add(Embedding(output_dim =32, input_dim=3800, input_length=380))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()
scores=model.evaluate(x_test, y_test, verbose=1)
scores[1]

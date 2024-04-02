## 分子毒性预测任务

在药物发现的早期，快速且精准地评估血液毒性至关重要。而作为在多个领域都表现出色的数据驱动方法——深度学习，利用其进行分子性质的预测对于药物的开发有着重大的意义。
对于任务一，需要根据分子的SMILSE格式预测其对应的二分类任务。首先，对数据集进行了类别均衡的预处理，并把SMILE字符格式的化学式转化成分子图结构。通过传统机器学习和深度学习方法的性能比较，选用了一种端到端双头transformer神经网络用于分子性质的高精度预测。该神经网络框架引入了新的激活函数beaf，大大提高分子特征非线性表示的泛化能力；在分子编码部分加入残差网络，解决梯度爆炸问题，保证模型的快速收敛性；利用基于双头注意的transformer提取分子的内在细节特征，并合理分配权值，实现分子性质的高精度预测。同时，本文将神经网络提取出来的特征放入XGboost中，发现摩根分子指纹和XGboost的加入对模型的性能改进有一定影响。

![dhtnn.jpg](chemprop%2Fdhtnn.jpg)
对于任务二，需要根据分子的SMILES格式预测其对应的617个二分类任务。为了有效的解决标签缺失问题，在DHTNN模型的基础上结合Mean Teacher半监督学习算法对617个二分类任务进行预测。并同时对结合半监督学习和全监督学习的方法进行对比试验，比较它们对各性质的预测AUC和所有性质预测的平均AUC。

![mean teacher.jpg](chemprop%2Fssl%2Fmean%20teacher.jpg)

## 运行环境
python3.8
[requirements.txt](requirements.txt)
## 代码结构
bTox/rToxcast：数据集，包括训练，验证，测试
bTox_save/rToxcast_save：模型产生的结果和日志文件
chemprop：预测任务的核心代码文件夹，包括数据读取处理，特征提取，网络模型构造，参数定义，参数网格搜索，模型训练验证代码，模型解释代码等
其他：对结果的处理展示
train.py：模型训练的入口文件

## 训练和测试模型

train.py(chemprop_train()) --data_path bTox\train.csv --dataset_type classification --save_dir bTox_save
train.py(chemprop_interpret()) --data_path bTox\train.csv --dataset_type classification --checkpoint_dir bTox_save
```

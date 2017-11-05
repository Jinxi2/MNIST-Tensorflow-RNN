import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #导入数据

lr=0.001#学习效率
training_iters=500000 #训练次数
batch_size=128 #每次抓取的数据量

n_inputs=28  #28列
n_steps=28  #28行
n_hidden_unis=128  #隐藏层单元个数
n_classes=10  #0-9

#进行网络结构的建立

with tf.name_scope('inputs'):  #input
    x = tf.placeholder(tf.float32,[None,n_steps,n_inputs],name='x_inputs')  #得到传递进来的训练样本 图片
    y = tf.placeholder(tf.float32,[None,n_classes],name='y_inputs') #标签
    weights=\
    {
        'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis]),name='weights'), #从样本进行输入
        'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]),name='weights1') #输出至隐藏层
    }
    biases=\
    { #偏置
        'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,]),name='biases'), #(128,)
        'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]),name='biases1') #(10,)
    }

def RNN(X,weights,biases):#X（128batch,28steps,28inputs）
    with tf.name_scope('layer'):
        X = tf.reshape(X,[-1,n_inputs]) #重置为128*28,28的向量
        X_in = tf.matmul(X,weights['in'])+biases['in'] #X与weight矩阵乘，加偏置
        X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_unis]) #变换形式，置为n*28*128向量
        tf.summary.histogram('X_in', X_in) #储存结果
        #cell
    with tf.name_scope('cell'):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unis,forget_bias=1.0, state_is_tuple=True) #该层128个cell，偏置 = 1
        _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32) #初始化
        outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False) #输入至cell中
        tf.summary.histogram('outsputs', outputs) #储存输出和状态
        tf.summary.histogram('outsputs', states)
        #输出层
    with tf.name_scope('result'):
        result = tf.matmul(states[1],weights['out'])+biases['out'] #计算结果，将状态与权重（出）相乘，加偏置（出）
        tf.summary.histogram('result', result)
        return result

pred = RNN(x,weights,biases) #调用RNN进行预测

with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred),name='loss') #求平均值 交叉熵（训练成本）cost
    tf.summary.scalar('loss', cost)
with tf.name_scope('train'):  #进行优化
    train_op = tf.train.AdamOptimizer(lr).minimize(cost) #以lr为学习速度最小化cost   ？？？？？？？？？？？

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1)) #比较预测结果和正确答案是否相等，结果为向量
accurary = tf.reduce_mean(tf.cast(correct_pred,tf.float32)) #将correct_pred转化为float32类型，求均值
tf.summary.scalar('accurary', accurary)
init = tf.global_variables_initializer()
saver = tf.train.Saver() #保存模型参数
with tf.Session() as sess: #将sess指向要处理的地方
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs1/", sess.graph) #打开写文件指针，准备记录
    sess.run(init) #激活，开始训练
    step=0
    while step<10000:
                    # step*batch_size<training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size) #再取下一个batch块，返回image和label
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs]) #同之前操作
        sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})
        if step%100==0: #每100步，进行一次记录
            res = sess.run(merged,feed_dict={x:batch_xs,y:batch_ys}) #管理数据
            writer.add_summary(res,step) #记录当前步数和数据
            print(sess.run(accurary,feed_dict={x:batch_xs,y:batch_ys})) #正确率
        step += 1
    save_path = saver.save(sess, "my_net1\save_net.ckpt")
import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #导入已下载数据

batch_size = 128

n_inputs=28#mnist数据集是28*28像素，n_inputs代表的是每一行的28列
n_steps=28#代表28行
n_hidden_unis=128#隐藏层单元个数
n_classes=10#分类称10个，代表0-9
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32,[None,n_steps,n_inputs],name='x_inputs')  #得到传递进来的训练样本 图片  ？？？？？？格式？？？
    y = tf.placeholder(tf.float32,[None,n_classes],name='y_inputs') #标签
    weights={
        'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis]),name='weights'),
        'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]),name='weights1')
    }
    biases={
    #(128,)
        'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,]),name='biases'),
    #(10,)
        'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]),name='biases1')
    }

def RNN(X,weights,biases):#X（128batch,28steps,,28inputs）128个数字，每个都是28行28列
    ##隐藏层
    #X(128*28,28)
    with tf.name_scope('layer'):
        X=tf.reshape(X,[-1,n_inputs])
        #X(128*28,28)
        X_in=tf.matmul(X,weights['in'])+biases['in']
        #X(128,28,28)
        X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_unis])
        tf.summary.histogram('X_in', X_in)
        #cell
    with tf.name_scope('cell'):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unis,forget_bias=1.0, state_is_tuple=True)
        _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
        outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)

        tf.summary.histogram('outsputs', outputs)
        tf.summary.histogram('outsputs', states)
        #输出层
    with tf.name_scope('result'):
        result=tf.matmul(states[1],weights['out'])+biases['out']
        tf.summary.histogram('result', result)
        return result
pred=RNN(x,weights,biases)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1)) #比较预测结果和正确答案是否相等，结果为向量
accurary = tf.reduce_mean(tf.cast(correct_pred,tf.float32)) #将correct_pred转化为float32类型，求均值
tf.summary.scalar('accurary', accurary)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "my_net1/save_net.ckpt")  #读取模型

    step=0
    while step*batch_size<10000:
        batch_xs,batch_ys = mnist.test.next_batch(batch_size) #再取下一个batch块，返回image和label
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs]) #同之前操作
        if step%10==0: #每10步，进行一次记录
            print(sess.run(accurary,feed_dict={x:batch_xs,y:batch_ys})) #正确率
        step += 1
import tensorflow as tf

t,f = 1. ,-1.
b = 1.
train_in = [[f,f,b],[t,f,b],[f,t,b],[t,t,b]]
train_out = [[f],[f],[f],[t]]

w = tf.Variable(tf.random_normal([3,1]))

def step(x):
    is_greater = tf.greater(x,0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float,2)
    return tf.subtract(doubled,1)

output = step(tf.matmul(train_in,w))
error = tf.subtract(train_out,output)
mse = tf.reduce_mean(tf.s quare(error))

delta = tf.matmul(train_in,error,transpose_a = True)
train = tf.assign(w,tf.add(w,delta))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

err, target = 1,0

#epoch is nothing but the number of cycles or iteration to get to the minimum result

epoch, max_epoch = 0,10
while err>target and epoch<max_epoch:
    epoch +=1
    err,_ = sess.run([mse,train])
    print('epoch:',epoch,'mse:',err)




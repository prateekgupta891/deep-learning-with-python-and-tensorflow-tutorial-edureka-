import tensorflow as tf

from tensorflow.exapmles.tutorials.minst from input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,shape=[None,784])
#None means we are not putting any restrictions on the shape
y = tf.placeholder(tf.float32,shape=[None,10])

w = tf.Variable(tf.zeroes([784,10])) #784x10 matrix
b = tf.Variable(tf.zeroes([10])) #10 classes

sess.run(tf.global_variables_initializer())
y = tf.matmul(x,w) + b

#definig the loss by training across all examples
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
#label(target ouput) logits(actaul output) softmax find out difference between them
#summing up individual error to get final error.

#gradient descent

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch[0],y:batch[1]})

correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels}))





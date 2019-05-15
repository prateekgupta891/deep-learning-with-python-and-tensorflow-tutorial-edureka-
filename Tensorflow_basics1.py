import tensorflow as tf

#using constant tensors
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)
sess = tf.Session()
#runnung a computational graph
print(sess.run([node1,node2]))

#always print constant values

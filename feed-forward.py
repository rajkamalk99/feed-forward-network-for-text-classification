

'''
input > weights > hidden layers1(activation function) > weights > hidden layer2(activation function) > weights > output layer - forward propagation

compare utput to the intendeed output > cost function

optimization function (optimizer) > minimize the cost (AdamOptimizer.....SGD)

backpropagaion - optimizer goes back and adjust the weights to minimize the cost function 

forward propagation + backpropagation = epoch 

'''

from data_preprocessing import create_train_test_data, create_lexicon
import tensorflow as tf
import numpy as np

true_file = "/home/raj/Documents/AI/fake_news/datasets/true_reduced5.txt"
fake_file = "/home/raj/Documents/AI/fake_news/datasets/fake_reduced5.txt"
true_dummy = "/home/raj/Documents/AI/fake_news/true_dummy.txt"
fake_dummy = "/home/raj/Documents/AI/fake_news/fake_dummy.txt"

with open(true_file, "r") as f1:
    true_contents = f1.readlines()
with open(fake_file, "r") as f2:
    fake_contents = f2.readlines()
print("read both the files")
count = 0
train_x = []
train_y = []
test_x = []
test_y = []
lexicon = create_lexicon(true_contents, fake_contents)
print("lexicon created")
for true_line, fake_line in zip(true_contents, fake_contents):
    count += 1
    if count == 1:
        f1 = open(true_dummy, "w+")
        f2 = open(fake_dummy, "w+")
        f1.write(str(true_line)+"\n")
        f2.write(str(fake_line)+"\n")
    elif count > 1 and count < 250:
        f1.write(str(true_line)+"\n")
        f2.write(str(fake_line)+"\n")
    elif count == 250:
        f1.write(str(true_line)+"\n")
        f2.write(str(fake_line)+"\n")
        train_x_, train_y_, test_x_, test_y_ = create_train_test_data(true_dummy, fake_dummy, lexicon)
        train_x += train_x_
        train_y += train_y_
        test_x += test_x_
        test_y += test_y_
        f1.close()
        f2.close()
        count = 0

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 2
batch_size = 100

x = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data):

    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}

    # input_data * weights + biases > activation_function

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l1, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 30

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([cost, optimizer], feed_dict={x:batch_x, y:batch_y})
                if c != None:
                    epoch_loss += c
                i += batch_size                                 
            print("Epoch ", epoch+1, " completed out of ", hm_epochs, " with loss ", epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)

from python.util import *
from python.classification_helper import Data_Helper

train_data_dir = './data/sentance_classification'
vocab_list_file= './data/word_list.txt'

                        #[ (fitter_size, kernal_size)] 
cnn_fitter_kernal_size = [ (12, 5) ]
fc_layer_size          = [ 32 ]

learning_rate = 1e-3
embedding_dim = 64
seq_length    = 12
helper        = Data_Helper(train_data_dir, vocab_list_file)
vocab_size    = helper.get_vocab_size()
class_count   = helper.get_class_count()

input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
input_y = tf.placeholder(tf.int64, [None], name='input_y')
dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

#transpose_y = tf.transpose(input_y)

y           = input_y
#adverb_y    = transpose_y[1]
#object_y    = transpose_y[2]
#number_y    = transpose_y[3]

with tf.device('/cpu:0'):
    embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
    embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

cnn_model = make_cnn_model(embedding_inputs, cnn_fitter_kernal_size, fc_layer_size, dropout_prob)

pred_vec      = tf.layers.dense(cnn_model, class_count, name='pred_vector')
adverb_vec    = tf.layers.dense(cnn_model, class_count, name='pred_adverb_vector')
object_vec    = tf.layers.dense(cnn_model, class_count, name='pred_object_vector')
number_vec    = tf.layers.dense(cnn_model, class_count, name='pred_number_vector')

pred_class    = tf.argmax(tf.nn.softmax(pred_vec), 1, name='pred_class')
#adverb_class  = tf.argmax(tf.nn.softmax(adverb_vec), 1, name='adverb_class')
#object_class  = tf.argmax(tf.nn.softmax(object_vec), 1, name='object_class')
#number_class  = tf.argmax(tf.nn.softmax(number_vec), 1, name='number_class')

loss          = make_loss(pred_vec, y, class_count, name='loss')
#adverb_loss   = make_loss(adverb_vec, adverb_y, class_count, name='adverb_loss')
#object_loss   = make_loss(object_vec, object_y, class_count, name='object_loss')
#number_loss   = make_loss(number_vec, number_y, class_count, name='number_loss')

acc_rate      = make_acc_rate(pred_class, y, name='acc_rate')
#adverb_rate   = make_acc_rate(adverb_class, adverb_y, name='adverb_loss')
#object_rate   = make_acc_rate(object_class, object_y, name='object_loss')
#number_rate   = make_acc_rate(number_class, number_y, name='number_loss')

global_loss   = loss# + adverb_loss + object_loss + number_loss

opt           = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(global_loss, name='optimizer')

global_step   = tf.train.get_or_create_global_step()

increment_global_step = tf.assign(global_step, global_step + 1)

init = tf.global_variables_initializer()
sess = tf.Session()
#with tf.Session() as sess:

sess.run(init)

def train_model(iter_size=1000, batch_size=100, show_step=10, test_step=100):
    for i in range(iter_size):
        train_data, label_data = helper.get_data(batch_size, pad_sequence_length=seq_length)
        out = sess.run([global_loss, acc_rate, increment_global_step, opt], { input_x : train_data, input_y: label_data, dropout_prob : 0.5})
        if out[2] % show_step == 0:
                tf.logging.info('Step #%d: accuracy %.2f%%, cross entropy %f' %
                    (out[2], out[1] * 100, out[0]))


def test_model(test_list):
    source_data, processed_data = helper.string_to_input_data(test_list, pad_sequence_length=seq_length)
    out_data = sess.run(pred_class, { input_x : processed_data, dropout_prob : 1})
    return helper.get_class_name(int(out_data))

train_model()

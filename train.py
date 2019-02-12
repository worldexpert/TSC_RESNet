from utils import *
import time
from datetime import datetime
#import pandas as pd
#import itertools



num_residual_blocks = 5
batch_size = 256

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('version', 'new2', '''A version number defining the directory to save
logs and checkpoints''')
tf.app.flags.DEFINE_string('optimizer', 'adam', '''Opimizer : AdamOptimizer/ MomentumOptimizer / RMSPropOptimizer''')

train_dir = './model_' + FLAGS.version + FLAGS.optimizer
log_dir = './log_' + FLAGS.version + FLAGS.optimizer


class Train:
    def __init__(self) -> None:
        self.placeholders()

    def placeholders(self) -> None:
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 1])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=(None))

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 1])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=(None))

        self.learningrate_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
        self.istraining = tf.placeholder(dtype=tf.bool, shape=[])
        self.resize = tf.placeholder(dtype=tf.int32, shape=(None))

    '''
    learning rate decay : 
    '''

    def build_train_validation_graph(self, opt, global_step):
        '''
        This function builds the train graph and validation graph at the same time.

        '''

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph

        #model = []
        with tf.variable_scope('conv0'):
            #in_channel = self.image_placeholder.get_shape().as_list()[-1]
            #bn_layer = batch_normalization_layer(self.image_placeholder, in_channel)

            bn_layer= tf.layers.batch_normalization(resizeImageData(self.image_placeholder, self.resize), training=self.istraining)
            conv0 = tf.layers.conv2d(bn_layer, 16, [3, 3], activation= tf.nn.relu, padding='SAME')
            activation_summary(conv0)

        for i in range(num_residual_blocks):
            with tf.variable_scope('conv1_%d' % i):
                if i == 0:
                    conv1 = residual_block(conv0, 16, self.istraining, first_block=True)
                else:
                    conv1 = residual_block(conv1, 16, self.istraining)
                activation_summary(conv1)


        for i in range(num_residual_blocks):
            with tf.variable_scope('conv2_%d' % i):
                conv2 = residual_block(conv1, 32, self.istraining)
                activation_summary(conv2)

        for i in range(num_residual_blocks):
            with tf.variable_scope('conv3_%d' % i):
                conv3 = residual_block(conv2, 64, self.istraining)
            assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc1'): #fully connected layer
            #in_channel = conv3.get_shape().as_list()[-1]
            #bn_layer = batch_normalization_layer(conv3, in_channel)
            bn_layer= tf.layers.batch_normalization(conv3, training=self.istraining)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])
            #global_pool = tf.contrib.layers.flatten(relu_layer)
            fc1 = tf.layers.dense(global_pool, 120, activation= tf.nn.relu, name='fc1')

        with tf.variable_scope('fc2'):  # fully connected layer

            #assert global_pool.get_shape().as_list()[-1:] == [64]
            model = tf.layers.dense(fc1, num_class, activation= None, name='model') # in 43(num_class) classes


        # one hot encode ??

        one_hot_y = tf.one_hot(self.label_placeholder, 43)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=one_hot_y))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            optimizers= {'adam': tf.train.AdamOptimizer(learning_rate=self.learningrate_placeholder).minimize(cost,global_step=global_step),
                'mome': tf.train.MomentumOptimizer(learning_rate=self.learningrate_placeholder, momentum=0.9).minimize(cost,global_step=global_step),
                'rmsp': tf.train.RMSPropOptimizer(learning_rate=self.learningrate_placeholder, momentum=0.9).minimize(cost,global_step=global_step)}
            optimizer = optimizers.get(opt)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss


        return optimizer, cost, model


    def train(self, init_lr, EPOCHS = 30, opt = 'adam') :

        training_data, training_labels = read_training_data()
        test_data, test_labels = read_validation_data()

        #print('training_data.shape : ', training_data.shape)
        #print('training_labels.shape : ', training_labels.shape)

        #print('test_data.shape : ', test_data.shape)
        #print('test_labels.shape : ', test_labels.shape)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer, cost, model = self.build_train_validation_graph(opt, global_step)
        tf.summary.scalar('cost', cost)

        one_hot_y = tf.one_hot(self.vali_label_placeholder, 43)
        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(one_hot_y, 1) )
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        run_meta = tf.RunMetadata()
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restored from checkpoint...')
            else:
                sess.run(tf.global_variables_initializer())

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            #step_list = []
            #train_error_list = []
            #val_error_list = []

            BATCH_SIZE = 256
            from sklearn.utils import shuffle

            X_train, y_train = preProcessingData(training_data, training_labels)
            X_test, y_test = preProcessingData(test_data, test_labels)

            for step in range(EPOCHS):  #
                report_freq= 500

#                train_batch_data, train_batch_labels = generate_augment_train_batch(training_data, training_labels, batch_size)


                num_examples = len(X_train)

                X_tref, y_tref = shuffle(X_train, y_train)

                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_tref[offset:end], y_tref[offset:end]

                    start_time = time.time()

                    _, train_loss_value = sess.run([optimizer, cost],
                                                        feed_dict={self.image_placeholder: batch_x,
                                                        self.label_placeholder: batch_y,
                                                        self.learningrate_placeholder: init_lr,
                                                        self.istraining : True })
                    duration = time.time() - start_time



                summary_str = sess.run(summary_op, feed_dict={self.image_placeholder: batch_x,
                                                 self.label_placeholder: batch_y,
                                                 self.learningrate_placeholder: init_lr,
                                                self.istraining : True })
                summary_writer.add_summary(summary_str, global_step=sess.run(global_step))

                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(sess.run(global_step))
                print(format_str % (datetime.now(), train_loss_value, examples_per_sec,
                                    sec_per_batch))

                print('Validation loss = ', train_loss_value)
                print('----------------------------')

                if step == EPOCHS*0.5 : #or step == steps*0.8 :
                    init_lr = 0.1 * init_lr
                    print('Learning rate decayed to ', init_lr)


            # sess = tf.Session()
            total_accuracy = 0
            num_examples = len(X_test)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_test[offset:end], y_test[offset:end]

                acc = sess.run(accuracy, feed_dict={self.image_placeholder: batch_x,
                                                       self.vali_label_placeholder: batch_y,
                                                self.istraining : True })
                total_accuracy += (acc * len(batch_x))
            print('Accuracy :', total_accuracy / num_examples)


            saver.save(sess, train_dir+'/resnet.ckpt', global_step=global_step )
            if flops is not None:
                print('TF stats gives', flops.total_float_ops)


    #   def test(self):
            validation_step = tf.Variable(0, trainable=False, name='validation_step')
            #model = tf.get_variable("model", [1])


    def test_eachSize(self, opt = 'adam') :
        global_step = tf.Variable(0, trainable=False, name='global_step')
        test_data, test_labels = read_validation_data()

        optimizer, cost, model = self.build_train_validation_graph(opt, global_step)

        one_hot_y = tf.one_hot(self.vali_label_placeholder, 43)
        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(one_hot_y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        X_test = preProcessingImageData(test_data)
        y_test = test_labels

        for resize in [12, 16, 20, 24, 28, 32] :

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                saver = tf.train.Saver(tf.global_variables())

                ckpt = tf.train.get_checkpoint_state(train_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Restored from checkpoint...')
                else :
                    print('No checkpoint')
                    exit()

                BATCH_SIZE = 256

                # sess = tf.Session()
                total_accuracy = 0
                num_examples = len(X_test)
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_test[offset:end], y_test[offset:end]

                    acc = sess.run(accuracy, feed_dict={self.image_placeholder: batch_x,
                                                        self.vali_label_placeholder: batch_y,
                                                        self.istraining: True,
                                                        self.resize: resize})
                    total_accuracy += (acc * len(batch_x))
                print(resize, ' Accuracy :', total_accuracy / num_examples)

                '''
                labels = sess.run(model, feed_dict={self.image_placeholder: test_data,
                                                    self.istraining: False})
                '''

        #display_random_image(test_ori_data, labels)

    def test_accumulate(self, opt = 'adam'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        test_data, test_labels = read_validation_data()

        optimizer, cost, model = self.build_train_validation_graph(opt, global_step)

        one_hot_y = tf.one_hot(self.vali_label_placeholder, 43)
        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(one_hot_y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        X_test = preProcessingImageData(test_data)

        y_test = test_labels
        categorical_predics = np.zeros((len(test_data),43))
        ambigue_count = 0



        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())

            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restored from checkpoint...')
            else:
                print('No checkpoint')
                exit()

            BATCH_SIZE = 256
            for resize in [32]: #12, 16, 20, 24, 28,
                # sess = tf.Session()
                total_accuracy = 0
                num_examples = len(X_test)


                predics = np.array([]).reshape(0, 43) #len(test_data)

                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_test[offset:end], y_test[offset:end]

                    predic = sess.run(model, feed_dict={self.image_placeholder: batch_x,
                                                        self.vali_label_placeholder: batch_y,
                                                        self.istraining: True,
                                                        self.resize: resize})
                    predics = np.concatenate((predics, predic))

                    acc = sess.run(accuracy, feed_dict={self.image_placeholder: batch_x,
                                                        self.vali_label_placeholder: batch_y,
                                                        self.istraining: True,
                                                        self.resize: resize})
                    total_accuracy += (acc * len(batch_x))
                print(resize, ' Accuracy :', total_accuracy / num_examples)

                categorical_predics = categorical_predics + predics * resize * resize

        from scipy.special import softmax

        count = 0
        #wrong_images = np.array([]).reshape(0, 32, 32, 3)
        wrong_images = []
        wrong_predics = []
        wrong_labels = []
        for idx in range(predics.shape[0]) :
            if np.argmax(predics[idx]) != y_test[idx] :
                d = softmax(predics[idx])
                for i in range(43):
                    if (d[i] > 0.01):
                        print("[",i, "]:",d[i], end=',')
                print()
                count = count +1
                #print(wrong_images.shape, np.array(test_data[idx]).shape)

                wrong_images.append(test_data[idx])
                wrong_predics.append(np.argmax(predics[idx]))
                wrong_labels.append(y_test[idx])

        print("wrong inference count = ", count)
        display_random_image(wrong_images, wrong_predics, wrong_labels)


        np.set_printoptions(precision=4)

        for rate in [0.9, 0.8, 0.7, 0.6, .5]:
            for index in predics :
                d = softmax(index)
                ambigue = True
                for i in range(43) :
                    if (d[i] > rate) :
                        ambigue = False
                if (ambigue == True) :
                    ambigue_count = ambigue_count +1

            print("Total count < ", rate ," :  count =", ambigue_count, "percentage = ", ambigue_count / num_examples)

        print()

        d = softmax(predics[0])
        for i in range(43):
            if (d[i] < 0.001):
                print(0, end=',')
            else:
                print(d[i], end=',')

        for index in predics:
            count = 0
            d = softmax(index)
            for i in range(43):
                if (d[i] > 0.39):
                    count = count + 1
            if (count > 1):
                tmp = d

        for i in range(43):
            if (tmp[i] < 0.001):
                print(0 , end=',')
            else :
                print(tmp[i] , end=',')
        print()



        sum = 0
        for idx in range(num_examples):
            sum = sum + np.equal(np.argmax(categorical_predics[idx]), y_test[idx])


        print('Accumulated Accuracy :', sum / num_examples)


train = Train()
#train.train(0.01, EPOCHS = 30, opt = 'adam') #'adam', 'mome'

#train.test_eachSize()
train.test_accumulate()


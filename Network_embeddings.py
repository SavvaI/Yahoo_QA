from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import batch_generator



class EndToEndNetwork:
    def __init__(self, num_hops=1, batch_size=50, validation_size=300, sentence_embed_dim=200, num_variants = 5, valid_num_variants = 10):
        self.valid_num_variants = valid_num_variants
        self.sentence_output_embed_dim = self.sentence_input_embed_dim = self.question_embed_dim = self.memory_dim = sentence_embed_dim
        self.batch_generator = batch_generator.BatchGenerator(batch_size, num_variants=num_variants)
        self.num_variants = self.batch_generator.num_variants
        self.num_hops = num_hops
        self.vocabluary_size = len(self.batch_generator.encode_dict)
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.session = tf.Session()


    def init_variables(self):
        self.session.run(self.init)

    def build_graph(self):
        # Variables
        bb = self.bb = tf.Variable(tf.truncated_normal([1, self.question_embed_dim], stddev=0.001))
        ba = self.ba = tf.Variable(tf.truncated_normal([1, self.sentence_input_embed_dim], stddev=0.001))
        bc = self.bc = tf.Variable(tf.truncated_normal([1, self.sentence_output_embed_dim], stddev=0.001))
        A = self.A = []
        B = self.B = tf.Variable(tf.truncated_normal([1024, self.question_embed_dim], stddev=0.001))
        C = self.C = []
        for i in range(self.num_hops):
            self.A.append(
                tf.Variable(tf.truncated_normal([1024, self.sentence_input_embed_dim], stddev=0.001)))
            self.C.append(
                tf.Variable(tf.truncated_normal([1024, self.sentence_output_embed_dim], stddev=0.001)))
        self.init = tf.initialize_all_variables()

        # Batch Graph
        batch_question = tf.placeholder(tf.float32, [None, 1024])
        batch_sentences = tf.placeholder(tf.float32, [None, None, 1024])
        batch_y_ = tf.placeholder(tf.int32, [None])

        batch_y_one_hot = tf.one_hot(batch_y_, self.batch_generator.num_variants)
        global_step = tf.Variable(0, trainable=False)
        self.make_step = global_step.assign(global_step + 1)
        starter_learning_rate = 1.0
        learning_rate = starter_learning_rate
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 350, 0.5, staircase=False)
        sentences_reshaped = tf.reshape(
            batch_sentences,
            [self.batch_size * self.num_variants, -1])

        question_embed = tf.matmul(batch_question, B) + bb
        u = tf.reshape(
            tf.tile(question_embed, [1, self.num_variants]),
            [self.num_variants * self.batch_size, -1])

        for j in range(self.num_hops):
            sentences_output_embed = tf.matmul(sentences_reshaped, C[j]) + bc
            sentences_input_embed = tf.matmul(sentences_reshaped, A[j]) + ba

            p = tf.nn.softmax(
                tf.reshape(
                    tf.reduce_sum(
                        u * sentences_input_embed,
                        1
                    ),
                    [self.batch_size, -1]
                ),
                -1
            )

            u = u + sentences_output_embed * tf.reshape(
                p,
                [-1, 1]
            )
        y = p
        cross_entropy = -tf.reduce_sum(tf.cast(batch_y_one_hot, dtype=tf.float32) * tf.log(y)) / tf.constant(float(self.batch_size))
        # self.example = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), batch_y_), tf.float32))



        # self.train_step = tf.train.AdadeltaOptimizer(learning_rate=0.5).minimize(cross_entropy)
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cross_entropy)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # gvs = optimizer.compute_gradients(cross_entropy)
        # grads = tf.clip_by_global_norm([grad for grad, var in gvs], 40)
        # self.grad_before_clipping = grads[1]
        # self.grad_after_clipping = tf.clip_by_global_norm(grads[0], 1)[1]
        # capped_gvs = [(grads[0][i], gvs[i][1]) for i in
        #               range(len(gvs))]  # [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        # self.app_grads = optimizer.apply_gradients(capped_gvs)
        #
        # #        train_step = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)
        #


        # Validation Graph

        validation_question = tf.placeholder(tf.float32, [None, None])
        validation_sentences = tf.placeholder(tf.float32, [None, None, None])
        validation_y_ = tf.placeholder(tf.int32, [None])

        validation_y_one_hot = tf.one_hot(validation_y_, self.valid_num_variants)
        sentences_reshaped = tf.reshape(
            validation_sentences,
            [self.validation_size * self.valid_num_variants, -1])

        question_embed = tf.matmul(validation_question, B) + bb
        u = tf.reshape(
            tf.tile(question_embed, [1, self.valid_num_variants]),
            [self.valid_num_variants * self.validation_size, -1])

        for j in range(self.num_hops):
            sentences_output_embed = tf.matmul(sentences_reshaped, C[j]) + bc
            sentences_input_embed = tf.matmul(sentences_reshaped, A[j]) + ba

            p = tf.nn.softmax(
                tf.reshape(
                    tf.reduce_sum(
                        u * sentences_input_embed,
                        1
                    ),
                    [self.validation_size, -1]
                ),
                -1
            )

            u = u + sentences_output_embed * tf.reshape(
                p,
                [-1, 1]
            )
        y = p

        # correct_rate = tf.reduce_mean(
        #     tf.cast(
        #         tf.equal(
        #             tf.reshape(validation_y_, shape=[-1, 1]),
        #             tf.cast(tf.argmax(y, 1), tf.int32)
        #         ),
        #         tf.float32
        #
        #     )
        # )
        correct_rate = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), validation_y_), tf.float32))
        # correct_rate = (tf.argmax(y, 1), validation_y_)
        # correct_rate = -tf.reduce_sum(tf.cast(validation_y_one_hot, dtype=tf.float32) * tf.log(y)) / tf.constant(float(self.validation_size))

        # correct_rate = tf.zeros([])
        # predictions_list = []
        # support_list = []
        #
        # for i in range(self.validation_size):
        #     question = tf.reshape(validation_question[i, :], [1, 1024])
        #     sentences = validation_sentences[i, :, :]
        #     y_ = tf.reshape(validation_y_one_hot[i, :], [1, self.batch_generator.num_variants])
        #
        #     # question_embed = tf.reshape(
        #     #     tf.reduce_sum(
        #     #         tf.nn.embedding_lookup(B, question), 0),
        #     #     [1, self.question_embed_dim])
        #     question_embed = tf.matmul(question, B) + bb
        #
        #
        #     u = question_embed
        #
        #     for j in range(self.num_hops):
        #         # sentences_output_embed = tf.reduce_sum(
        #         #     tf.nn.embedding_lookup(C[j], sentences), 1)
        #         # sentences_input_embed = tf.reduce_sum(
        #         #     tf.nn.embedding_lookup(A[j], sentences), 1)
        #         sentences_output_embed = tf.matmul(sentences, C[j]) + bc
        #         sentences_input_embed = tf.matmul(sentences, A[j]) + ba
        #
        #         self.dotproduct = tf.matmul(u, sentences_input_embed, transpose_b=True)
        #         p = tf.nn.softmax(
        #             tf.matmul(u, sentences_input_embed, transpose_b=True))
        #         support_list.append(tf.argmax(p, 1))
        #         u = u + tf.reduce_sum(
        #             tf.multiply(tf.transpose(p), sentences_output_embed), 0)
        #
        #     y = p
        #
        #     predictions_list.append(tf.argmax(y, 1))
        #
        # support = tf.reshape(
        #     tf.stack(support_list), [self.validation_size, self.num_hops])
        # predictions = tf.reshape(tf.stack(predictions_list), [-1])
        # correct_rate = tf.reduce_mean(
        #     tf.cast(
        #         tf.equal(
        #             predictions, tf.argmax(validation_y_one_hot, 1)),
        #         tf.float32))

        self.batch_placeholder = [batch_sentences, batch_question, batch_y_]
        self.validation_placeholder = [validation_sentences, validation_question, validation_y_]
        #        self.train_step = train_step
        self.correct_rate = correct_rate
        # self.predictions = predictions
        # self.support = support
        self.cross_entropy = cross_entropy
        self.y = y
        self.u = u
        self.init = tf.initialize_all_variables()

    def train(self, steps):
        for i in range(steps):
            self.batch_generator.get_text_batch(batch_type='train')
            batch = self.batch_generator.get_sum_batch(batch_type='train')
 #           batch = self.batch
            batch_dic = {self.batch_placeholder[0]: batch[0], self.batch_placeholder[1]: batch[1],
                         self.batch_placeholder[2]: batch[2]}

            if i % 100 == 0:
            #     print(self.session.run(self.example, feed_dict=batch_dic))
                print(self.session.run(self.cross_entropy, feed_dict=batch_dic))
            # self.session.run(self.app_grads, feed_dict=batch_dic)
            # self.session.run(self.make_step, feed_dict=batch_dic)
            self.session.run(self.train_step, feed_dict=batch_dic)

    def validate(self, print_examples, print_score=True, verbose = True):
        validation_set = self.batch_generator.get_sum_batch(batch_size=self.validation_size, num_variants=self.valid_num_variants, batch_type='validation')
#        validation_set = self.batch
        validation_dict = {self.validation_placeholder[0]: validation_set[0],
                           self.validation_placeholder[1]: validation_set[1],
                           self.validation_placeholder[2]: validation_set[2]}
        if print_score == True:
            print("Correct rate:  ", self.session.run(self.correct_rate, feed_dict=validation_dict))
        # ans = self.session.run(self.predictions, feed_dict=validation_dict)
        # # sup = self.session.run(self.support, feed_dict=validation_dict)
        # y = self.session.run(self.dotproduct, feed_dict=validation_dict)
        text = self.batch_generator.get_text_batch(encode=False, batch_size=self.validation_size, batch_type='validation')
        if verbose:
            for x in np.random.random_integers(0, self.validation_size - 1, print_examples):
                print(text[1][x][validation_set[2][x]], text[0][x])
                # preprocessing.print_task(validation_set[0][[x], :, :], validation_set[1][[x], :], validation_set[2][[x], :],
                #                          self.batch_gen.words)
                print("\n")
                print("Model's answer: ", end=" ")
                print(ans[x])
                # print("Support sentences:")
                # indicies = list(sup[x, :])
                # preprocessing.print_sentences(np.reshape(validation_set[0][x, indicies, :], [1, self.num_hops, -1]),
                #                               self.batch_gen.words)
                #
                # print("-------------------------")
            return self.session.run(self.correct_rate, feed_dict=validation_dict)

def train_yahoo():
    n = EndToEndNetwork()
    n.build_graph()
    n.init_variables()
    n.train(2000)
    n.validate(70, verbose=False)

train_yahoo()

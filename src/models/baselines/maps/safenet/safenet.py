import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
# import tensorflow.keras as keras
from keras.initializers import glorot_uniform
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
from tensorflow.keras.layers import Reshape, Input,Permute, Dropout, Add, Dense, Embedding, Conv2D, Layer, LSTM, BatchNormalization, Activation, MaxPooling2D, Flatten, LayerNormalization, AveragePooling2D,Bidirectional
# from keras_multi_head import MultiHeadAttention
import tensorboard as tb
from datetime import date,datetime,timedelta
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import math

try:
    os.chdir('project/2021eqs_predict')
except FileNotFoundError:
    print('___________________we now localed in :',os.getcwd())
# from tf_metrics_master import tf_metrics
import warnings
import random
warnings.filterwarnings("ignore")



class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded



class CNTmodel(object):
    def __init__(self, png_w=50,png_h=50,concat_years=10,
                eqs_feature_dim=22,epochs=3,learning_rate=1e-4,transformer_layers = 1, dense_num=2,
                num_heads = 2,projection_dim = 32,batch_size=16,drop_out=0.2,
                average='weighted',pos_indices=None,num_classes=6,goal_du=4,learning_rate_decay=False,
                goal='try_lr',positiontrainable=True,extendedfeature=False,pretrain_selfunknown=False,
                newdots=False,extendedXL_feature=False,usemap=True,newlunar=False,denselayer=[32,32,32,32],
                singlepatch=False,singlepatchname='',wsize=365,attention_mask=True,masktype='cos',scalermode='maxmin',finetuneyears=20,
                adjust_classsidelist=[1,1,1,1],rootdir='/data/my_dataset'):
        self.png_w=png_w
        self.png_h=png_h
        
        self.concat_years=concat_years
        self.eqs_feature_dim=eqs_feature_dim    #22,78,282
        self.goal_du=goal_du
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.transformer_layers=transformer_layers
        self.num_heads=num_heads
        self.projection_dim=projection_dim
        self.batch_size=batch_size
        self.drop_out=drop_out
        self.average=average
        self.pos_indices=pos_indices
        self.num_classes=num_classes
        if self.goal_du==4:
            self.num_patches=85 #120total 15*8
        elif self.goal_du==2:
            self.num_patches=326 #450 total 30*15
        self.num_patches+=1 ##add global token
        self.extendedfeature=extendedfeature        #加入paper8且只有频次和b值扩展到阅读的特征 78维
        self.extendedXL_feature=extendedXL_feature #加入paper8且各维度均扩展到月度的特征 282维
        self.transformer_units = [
            projection_dim * 2,
            projection_dim, ]
        self.mlp_head_units = [512, 256]
        self.usemap=usemap
        if self.usemap:self.png_channels=5
        else:self.png_channels=4    #去掉地震灰度图
        self.png_shape = [None,self.num_patches*self.concat_years,self.png_h,self.png_w,self.png_channels]
        self.goal=goal+'/lr_%s'%self.learning_rate
        self.dense_num=dense_num
        self.learning_rate_decay=learning_rate_decay
        self.positiontrainable=positiontrainable
        self.selfunknown=pretrain_selfunknown
        self.newdots= newdots
        self.newlunar = newlunar
        self.denselayer = denselayer
        self.wsize=wsize
        self.singlepatch=singlepatch
        self.singlepatchname=singlepatchname
        if attention_mask :
            self.attention_mask=pd.read_csv(os.path.join(rootdir,'attention_mask_%s.csv'%masktype),index_col=0).iloc[:,4:].values
            self.attention_mask=np.concatenate([np.ones((1,85)),self.attention_mask],axis=0)
            self.attention_mask=np.concatenate([np.ones((86,1)),self.attention_mask],axis=1)
            print(self.attention_mask.shape)

        else:
            self.attention_mask=None
        self.scalermode=scalermode
        self.finetuneyears=finetuneyears
        print('we use last %d years to finetune'%self.finetuneyears)
        self.adjust_classsidelist=adjust_classsidelist
        self.rootdir = rootdir
        self._init_graph()

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8 
        return tf.Session(config=config)

    def print_diff_mag_evaluation(self,num_classes,each_precision,each_recall,each_f1):
        if num_classes==6:
            print('0=<M<5:    precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[0],each_recall[0],each_f1[0]))
            print('5=<M<5.5:  precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[1],each_recall[1],each_f1[1]))
            print('5.5=<M<6:  precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[2],each_recall[2],each_f1[2]))
            print('6=<M<6.5:  precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[3],each_recall[3],each_f1[3]))
            print('6.5=<M<7:  precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[4],each_recall[4],each_f1[4]))
            print('7=<M:      precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[5],each_recall[5],each_f1[5]))
        elif num_classes==4:
            print('0=<M<5:    precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[0],each_recall[0],each_f1[0]))
            print('5=<M<6:  precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[1],each_recall[1],each_f1[1]))
            print('6=<M<7:  precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[2],each_recall[2],each_f1[2]))
            print('7=<M:      precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[3],each_recall[3],each_f1[3]))
        elif num_classes==2:
            print('0=<M<5:    precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[0],each_recall[0],each_f1[0]))
            print('5=<M:  precision: %.4f   recall:  %.4f   f1:  %.4f'%(each_precision[1],each_recall[1],each_f1[1]))
        else:
            raise ValueError('check your num_classes')

    def Evaluation(self,y_t, y_p,num_classes,multiclass=True,patch_mode='all'):
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
            precision_recall_fscore_support, confusion_matrix, mean_absolute_error
        if len(np.array(y_p).shape) >= 2:
            y_p = np.array(y_p).flatten()
        if len(np.array(y_t).shape) >= 2:
            y_t = np.array(y_t).flatten()
        # print('\ny_predition:\n', Counter(y_p))
        # print('y_true:\n', Counter(y_t))
        print(confusion_matrix(y_pred=y_p, y_true=y_t))
        each_mag_list_main, each_mag_list_up5 = [], []
        if multiclass:
            p_w = precision_score(y_t, y_p, average='weighted')
            r_w = recall_score(y_t, y_p, average='weighted')
            p_ma = precision_score(y_t, y_p, average='macro')
            r_ma = recall_score(y_t, y_p, average='macro')
            ac = accuracy_score(y_t, y_p)
            f1_w = f1_score(y_t, y_p, average='weighted')
            f1_ma = f1_score(y_t, y_p, average='macro')
            each_precision, each_recall, each_f1, _support = precision_recall_fscore_support(y_t, y_p, average=None,
                                                                                             labels=list(
                                                                                                 range(num_classes)))
            # self.print_diff_mag_evaluation(num_classes=num_classes, each_precision=each_precision,
                                        #    each_recall=each_recall, each_f1=each_f1)
            each_mag_list_main = self.eval_for_paper_format(acc=ac, macro_f1=f1_ma, each_precision=each_precision,
                                                            each_recall=each_recall, each_f1=each_f1)
            mae = mean_absolute_error(y_true=y_t, y_pred=y_p)
            y_5p = np.array(np.greater_equal(y_p, 1), 'int32')
            y_5t = np.array(np.greater_equal(y_t, 1), 'int32')
          
            each_mag_list_up5 = [accuracy_score(y_5t, y_5p), precision_score(y_5t, y_5p),
                                 recall_score(y_5t, y_5p), f1_score(y_5t, y_5p),
                                 f1_score(y_5t, y_5p, average='weighted')]
        
        return {'main': each_mag_list_main, 'up5': each_mag_list_up5}

    def eval_for_paper_format(self,acc,macro_f1,each_precision,each_recall,each_f1):
        each_mag_list=[]
        each_mag_list.append(acc)
        each_mag_list.append(macro_f1)
        for i in range(self.num_classes):
            for ix in [each_precision,each_recall,each_f1]:
                each_mag_list.append(ix[i])
        return each_mag_list

    def positional_encoding(self,inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
        '''Sinusoidal Positional_Encoding.

        Args:
        inputs: A 2d Tensor with shape of (N, T).
        num_units: Output dimensionality
        zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
        scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
            A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
        '''

        # N = self.batch_size
        validortest=tf.cond(tf.equal(self.is_train_valid_test,tf.constant(1)),lambda:10,lambda:1)
        N = tf.cond(tf.equal(self.is_train_valid_test,tf.constant(0)),lambda:self.batch_size,lambda:validortest)
        T = inputs.get_shape().as_list()[1]
        with tf.variable_scope(scope, reuse=reuse):
            new=tf.zeros_like(inputs)
            
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
            
            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
                for pos in range(T)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(position_enc,dtype= tf.float32)

            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                        lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
            # print('position_embedding shape:',outputs.shape)
            if scale:
                outputs = outputs * num_units ** 0.5
            # print('posional_encoding done')
            return outputs

    def mlp(self,x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = Dense(units, activation=tf.nn.relu)(x)
            x = Dropout(dropout_rate)(x)
        return x


    def get_shape_list(self,tensor, expected_rank=None, name=None):
        """Returns a list of the shape of tensor, preferring static dimensions.
        Args:
            tensor: A tf.Tensor object to find the shape of.
            expected_rank: (optional) int. The expected rank of `tensor`. If this is
            specified and the `tensor` has a different rank, and exception will be
            thrown.
            name: Optional name of the tensor for the error message.
        Returns:
            A list of dimensions of the shape of tensor. All static dimensions will
            be returned as python integers, and dynamic dimensions will be returned
            as tf.Tensor scalars.
        """
        if name is None:
            name = tensor.name

        if expected_rank is not None:
            assert_rank(tensor, expected_rank, name)

        shape = tensor.shape.as_list()

        non_static_indexes = []
        for (index, dim) in enumerate(shape):
            if dim is None:
                non_static_indexes.append(index)

        if not non_static_indexes:
            return shape

        dyn_shape = tf.shape(tensor)
        for index in non_static_indexes:
            shape[index] = dyn_shape[index]
        return shape

    def create_attention_mask_from_input_mask(self, to_mask, mask_broadcast):
        """Create 3D attention mask from a 2D tensor mask.
        Args:
            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
            to_mask: int32 Tensor of shape [batch_size, to_seq_length].
            mask_broadcast: tf.placeholder(tf.float32, shape=[None, self.num_patches,1])
        Returns:
            float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        
        to_mask = tf.cast(
            tf.reshape(to_mask, [-1, 1, self.num_patches]), tf.float32)

        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        #
        # `broadcast_ones` = [batch_size, from_seq_length, 1]
        
        broadcast_ones = tf.ones_like(mask_broadcast)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask


    def multihead_attention(self,queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None,attention_mask=None):
        '''Applies multihead attention.

        Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        dropout_rate: A floating point number.
        is_training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. If true, units that reference the future are masked.
        num_heads: An int. Number of heads.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
        A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                # print(queries.shape)
                num_units = queries.get_shape().as_list()[-1]
            #不设置则默认和queries最后一维的units大小即C_q一样
                # Set the fall back option for num_units

            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            # del Q_, K_,Q,K,V
            # gc.collect()
            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)

            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
            # gc.collect()
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)   #给一个近0的非0值
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            if attention_mask is not None:
                outputs=tf.multiply(outputs,tf.convert_to_tensor(attention_mask,tf.float32))
            
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!2019.3.14日新加的
            attention_score = outputs
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = Dropout(rate=dropout_rate)(outputs)

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
            # print(outputs.shape)
            # outputs = tf.layers.dense(outputs, hp.hidden_units, activation=tf.nn.relu)  # (N, T_k, C)
            # Residual connection
            outputs += queries

            # Normalize
            outputs = LayerNormalization(epsilon=1e-8)(outputs)  # (N, T_q, C)
            # gc.collect()
        return outputs,attention_score 
        # return outputs

    def identity_block(self,X, f, filters, stage, block):
        '''
        Resnet的标准模块
        :param X: 输入
        :param f: 卷积核大小
        :param filters: list，如【32，64，128】
        :param stage: str 用于命名
        :param block: str 用于命名
        :return:
        '''
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters

        X_shortcut = X

        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X

    def convolutional_block(self,X, f, filters, stage, block, s=2):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters

        X_shortcut = X

        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '2a',
                kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c',
                kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X


    def make_hparam_string(self):
        transformer_layers_name=str(self.transformer_layers)+'l'
        num_heads_name=str(self.num_heads)+'h'
        projection_dim_name=str(self.projection_dim)+'d'
        batch_size_name='bs'+str(self.batch_size)
        drop_out_name='dp'+str(self.drop_out)
        return 'lr_%.0E_%s_%s_%s_%s_%s' % (self.learning_rate, transformer_layers_name, num_heads_name,projection_dim_name,batch_size_name,drop_out_name)

   
    def _init_graph(self):
        
        self.x_png = tf.placeholder('float32',shape=[None,(self.num_patches-1)*self.concat_years,self.png_h,self.png_w,self.png_channels])
        self.x_eqs = tf.placeholder('float32',shape=[None,self.png_shape[1] // self.num_patches, self.num_patches, self.eqs_feature_dim])
        self.y = tf.placeholder(tf.int64,shape=[None,self.num_patches-1,1])
        self.drop_out_in_graph = tf.placeholder('float32')
        self.is_train_valid_test = tf.placeholder('int32')
        self.selfunknown_tf = tf.placeholder('bool')

        self.x = tf.reshape(self.x_png, (-1, self.png_shape[2], self.png_shape[3], self.png_shape[4])) #(batch_size*self.num_patches*concat_years, 50, 50, 5)
        self.x = Conv2D(32, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=42))(self.x)
        # print('after 1cnn shape is',self.x.shape)
        
        self.x = BatchNormalization(axis=3, name='bn_conv1')(self.x)
        self.x = Activation('relu')(self.x)
        # self.x = MaxPooling2D((3, 3), strides=(1, 1))(self.x)     ## try_4cnn
        self.x = MaxPooling2D((3, 3), strides=(2, 2))(self.x)   ##original
        # print('after maxpooling shape is',self.x.shape)

        self.x = self.convolutional_block(self.x, f=3, filters=[16, 16, 64], stage=2, block='a', s=1)
        self.x = self.identity_block(self.x, f = 3, filters = [16, 16, 64], stage = 2, block = 'b')
        self.x = self.identity_block(self.x, f = 3, filters = [16, 16, 64], stage = 2, block = 'c')
        # print('after 2cnn shape is',self.x.shape)
        
        ##3cnn        
        # print(self.x.shape) #
        self.x = self.convolutional_block(self.x, f=3, filters=[32, 32, 128], stage=3, block='a', s=2)
        self.x = self.identity_block(self.x, 3, [32, 32, 128], stage=3, block='b')
        self.x = self.identity_block(self.x, 3, [32, 32, 128], stage=3, block='c')
        self.x = self.identity_block(self.x, 3, [32, 32, 128], stage=3, block='d')
        # print('after 3cnn shape is',self.x.shape)


        self.x = AveragePooling2D((2, 2), strides = (2,2))(self.x)
        print(self.x.shape)
    
        self.x = Flatten()(self.x)
        self.x = Dense(32, kernel_initializer=glorot_uniform(seed=42),activation='relu')(self.x)
        self.x =  tf.reshape(self.x, (-1, self.png_shape[1],32)) ##(batch_size,self.num_patches*concat_years,self.projection_dim//2)
        self.x = tf.reshape(self.x,(-1, self.concat_years, self.num_patches-1, 32))  
            
        
        for la in range(len(self.denselayer)):
            if la ==0:
                self.x_eqs_d =  Dense(self.denselayer[0], kernel_initializer=glorot_uniform(seed=42),activation=tf.nn.leaky_relu)(self.x_eqs)
                print('x_eqs after first dense:',self.x_eqs_d.shape)
            else:
                self.x_eqs_d =  Dense(self.denselayer[la], kernel_initializer=glorot_uniform(seed=42),activation=tf.nn.leaky_relu)(self.x_eqs_d)
                print(self.x_eqs_d.shape)
        
        self.x = tf.concat([self.x, self.x_eqs_d[:,:,1:,:]],axis=-1)  # 组合起来，对应起来 (batch_size, concat_years, self.num_patches, 64)
        self.x = LayerNormalization(epsilon=1e-6)(self.x)
        self.x_eqs_d = LayerNormalization(epsilon=1e-6)(self.x_eqs_d[:,:,0,:])
        print('globel token shape',self.x_eqs_d.shape)
        # self.x = tf.concat([self.x, self.x_eqs],axis=-1)  # 组合起来，对应起来 (batch_size, concat_years, self.num_patches, 32)
        print(self.x.shape)
        self.x = Permute((2, 1, 3))(self.x)       #(batch_size,  self.num_patches, concat_years,32)
        print('after permute:',self.x.shape)
        
        self.x =  tf.reshape(self.x, (-1, self.png_shape[1] // (self.num_patches-1), self.x.shape[-1])) ##把85个patch变成batchsize的维度
        
        print('before get into LSTM',self.x.shape)
      
        self.x = LSTM(self.projection_dim)(self.x)     ##(batch_size* self.num_patches, concat_years, 32)--->(batch_size* self.num_patches, projection_dim)
        self.x_eqs_d =LSTM(self.projection_dim)(tf.reshape(self.x_eqs_d,(-1,self.concat_years,self.x_eqs_d.shape[-1])))
        self.x =  tf.reshape(self.x, (-1, self.num_patches-1, self.projection_dim))  ##恢复85个patch  (batch_size, self.num_patches, projection_dim)
        self.x = tf.concat([tf.reshape(self.x_eqs_d, (-1, 1, self.projection_dim)),self.x],axis=1)
        
        print('after reshape before position:',self.x.shape)
        if self.positiontrainable:
            print('position trainable!')
            self.encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(self.x)
        else:
            print('before encoded',self.x.shape)

            self.encoded_patches = self.positional_encoding(self.x, self.projection_dim)
        self.encoded_patches = LayerNormalization(epsilon=1e-6)(self.encoded_patches)
        
        self.encoded_patches  = Add()([self.x, self.encoded_patches])
        print('before transformer:',self.encoded_patches.shape)
        self.attention_score_list=[]

        for _ in range(self.transformer_layers):
            with tf.variable_scope("num_blocks_{}".format(_)):
                # Layer normalization 1.
                # print('10:', K.int_shape(x1))
                self.x1 = LayerNormalization(epsilon=1e-6)(self.encoded_patches)

                # Create a multi-head attention layer.
                self.attention_output,self.attention_score = self.multihead_attention(queries=self.x1,
                                                        keys=self.x1,
                                                        num_units=self.projection_dim,
                                                        num_heads=self.num_heads,
                                                        dropout_rate=self.drop_out_in_graph,
                                                        causality=False,
                                                        attention_mask=self.attention_mask)
              
                # Skip connection 1.
                self.x2 = Add()([self.attention_output, self.encoded_patches])
                print('%d num_blocks_ x2 shape:'%(_),self.x2.shape)

                # Layer normalization 2.
                self.x3 = LayerNormalization(epsilon=1e-6)(self.x2)
                # print('13 Layer normalization 2:', K.int_shape(x3))

                # MLP.
                self.x3 = self.mlp(self.x3, hidden_units=self.transformer_units, dropout_rate=self.drop_out_in_graph)
                # Skip connection 2.
                # print('the shape of x3 and x2 is :',K.int_shape(x3),K.int_shape(x2))
                print('%d num_blocks_ x3 shape:'%(_),self.x3.shape)
                
                self.encoded_patches = Add()([self.x3, self.x2])
                print('%d num_blocks_ encoded_patches shape:'%(_),self.encoded_patches.shape)

                self.attention_score_list.append(self.attention_score)
                # print('16 the encoded_patches shape is', K.int_shape(encoded_patches))  (?, 85+1, 512)
        
        self.logits = Dense(self.num_classes,kernel_initializer=glorot_uniform(seed=42))(self.encoded_patches)
        
        self.logits = self.logits[:,1:,:]
        ####调整logits的高震级权重
        # self.logits =self.logits*tf.constant([[1.0, 5, 51.0, 76.0]])
        self.pre = tf.reshape(tf.argmax(self.logits*tf.constant([self.adjust_classsidelist],dtype=tf.float32),-1),shape=(-1,1),name='mypre')
        self.logits = tf.reshape(self.logits,shape=(-1,self.num_classes),name='mylogits')
        self.prob = tf.nn.softmax(logits=self.logits)

        self.y_i = tf.reshape(self.y,shape=(-1,1))
        self.y_t = tf.reshape(self.y,shape=(-1,))
        print('logits shape:',self.logits.shape,' pre shape:',self.pre.shape,' y shape:',self.y_t.shape)
        if self.goal_du==4:
            if self.num_classes==6:
                self.class_weights =  tf.constant([[1.0, 7.0, 10.0, 20.0, 51.0, 76.0]])
            elif self.num_classes==4:
                self.class_weights =  tf.constant([[1.0, 4.0, 15.0, 78.0]])
            elif self.num_classes==2:
                self.class_weights =  tf.constant([[1.0, 3.0]])
        elif self.goal_du==2:
            if self.num_classes==6:
                self.class_weights =  tf.constant([[1.0, 23.0, 37.0, 86.0, 238.0, 367.0]])
            elif self.num_classes==4:
                self.class_weights =  tf.constant([[1.0, 14.0, 63.0, 367.0]])
            elif self.num_classes==2:
                self.class_weights =  tf.constant([[1.0, 11.0]])
        else : raise ValueError('Check your goal_du!')
     
        self.weights = tf.reduce_sum(self.class_weights * tf.one_hot(self.y_t,self.num_classes), axis=-1)
        self.loss = (tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_t))
        self.loss = tf.reduce_mean(self.loss*self.weights)
    
        begin_date,end_date=self.get_begin_and_end_date_for_split_dataset(madefor='train',concat_years=self.concat_years)
        total_size=(end_date-begin_date).days
        if self.learning_rate_decay:        
            self.g_step= tf.placeholder("int64")
            self.learning_rate2 = tf.train.exponential_decay(
                        learning_rate=self.learning_rate, global_step=self.g_step, decay_steps=total_size//self.batch_size, decay_rate=0.95, staircase=True)
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate2).minimize(self.loss)
        else:
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.sess = self._init_session()
        
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def get_begin_and_end_date_for_split_dataset(self,madefor,concat_years):
        if madefor=='train':
            begin_date=date(1971,1,1)+timedelta(days=365*(concat_years-1))
            end_date=date(2011,11,16)
        elif madefor=='valid':
            begin_date=date(2011,11,16)
            end_date=date(2020,11,16)
        elif madefor=='test':
            # begin_date=date(2020,11,16)
            begin_date=date(2021,11,15)
            end_date=date(2021,11,16)
        else:
            raise ValueError('check your key word')
        return begin_date,end_date
    
    def concat_many_years(self,count,concat_years,madefor,same_day=False):
        
        begin_date,end_date=self.get_begin_and_end_date_for_split_dataset(madefor=madefor,concat_years=concat_years)
        nowdate=begin_date+timedelta(days=count)
        # print('now is %s ' % str(nowdate))
        if nowdate<end_date:
            if concat_years is None:
                years = nowdate.year - 1971
            else:
                years = concat_years-1
            years_data=[]
            
            if same_day:
                for i in range(years, -1, -1):
                    if self.goal_du==4:
                        # print('concating---:\n now is %s ' % str(nowdate - timedelta(days=365 * i)))
                        with  open('/my_dataset/png_data/3png_concatdata_%s.pickle'%(date(nowdate.year-i,nowdate.month,nowdate.day)), 'rb') as f:
                            png_datas = pickle.load(f)
                    elif self.goal_du==2 and self.png_w==50:
                        with  open('/my_dataset/png_data/2du_3png_concatdata_%s.pickle'%(date(nowdate.year-i,nowdate.month,nowdate.day)), 'rb') as f:
                            png_datas = pickle.load(f)
                    elif self.goal_du==2 and self.png_w==25:
                        with  open('/my_dataset/png_data/newsize_2du_3png_concatdata_%s.pickle'%(date(nowdate.year-i,nowdate.month,nowdate.day)), 'rb') as f:
                            png_datas = pickle.load(f)
                    else: raise ValueError('check your goal_du !')
                    years_data.append(png_datas)

            else:
                for i in range(years, -1, -1):
                    if self.goal_du==4:
                       
                        with  open('/my_dataset/png_data/3png_concatdata_%s.pickle'%(nowdate - timedelta(days=365 * i)), 'rb') as f:
                            png_datas = pickle.load(f)
                    elif self.goal_du==2 and self.png_w==50:
                        with  open('/my_dataset/png_data/2du_3png_concatdata_%s.pickle'%(nowdate - timedelta(days=365 * i)), 'rb') as f:
                            png_datas = pickle.load(f)
                    elif self.goal_du==2 and self.png_w==25:
                        with  open('/my_dataset/png_data/newsize_2du_3png_concatdata_%s.pickle'%(nowdate - timedelta(days=365 * i)), 'rb') as f:
                            png_datas = pickle.load(f)
                    else: raise ValueError('check your goal_du !')
                    years_data.append(png_datas)
            
            years_data=np.array(years_data,dtype=np.float32)
            years_data=years_data.reshape(years_data.shape[1]*years_data.shape[0],years_data.shape[2],years_data.shape[3],years_data.shape[4])
            # print('years data shape:',years_data.shape)

            return np.array(years_data,dtype=np.float32)
        else:
            print('now is getting%s'%nowdate,'feature so we end \'%s\' dataset collecting'%madefor)

    def concat_many_years_same_day(self,month_dayall,count,concat_years,madefor):
        years_data=month_dayall[count:count+concat_years]  
        years_data=np.array(years_data,dtype=np.float32)
        years_data=years_data[:,:,:,:,:self.png_channels] 
        years_data=years_data.reshape(years_data.shape[1]*years_data.shape[0],years_data.shape[2],years_data.shape[3],years_data.shape[4])
        # print('years data shape:',years_data.shape)

        return np.array(years_data,dtype=np.float32)
        

    def generate_arrays_from_file(self,total_size,bs,concat_years,madefor):
        '''
        22维eqsdata存在一个train文件中，逐日现拼十年的png文件。same_day指代计算一年的方法是同日年份加一还是timedelta+day=365.
        '''

        global count,valid_count,test_count
        if self.goal_du==4:
           
            with  open('my_dataset/v2_eqs_data_and_labels/eqs_data_all_for_%s_%s.pickle'%(madefor,concat_years), 'rb') as f:    
                eqs_datas = pickle.load(f)
            print('for %s eqs_data shape is '%madefor,eqs_datas.shape)
            # with open('my_dataset/labels_mag_C_for_%s_%s.pickle'%(madefor,concat_years), 'rb') as f:     ##v1
            
            with open('my_dataset/v2_eqs_data_and_labels/labels_mag_C_class%s_for_%s_%s.pickle'%(self.num_classes,madefor,concat_years), 'rb') as f:
                    labels=pickle.load(f)
            print('for %s labels shape is :'%madefor,labels.shape)

        elif self.goal_du==2:
            with  open('my_dataset/2du_eqs_data_and_labels/eqs_data_all_for_%s_%s.pickle'%(madefor,concat_years), 'rb') as f:    
                eqs_datas = pickle.load(f)
            print('for %s eqs_data shape is '%madefor,eqs_datas.shape)
            # with open('my_dataset/labels_mag_C_for_%s_%s.pickle'%(madefor,concat_years), 'rb') as f:     ##v1
            with open('my_dataset/2du_eqs_data_and_labels/2du_labels_mag_C_class%s_for_%s_%s.pickle'%(self.num_classes,madefor,concat_years), 'rb') as f:
                labels=pickle.load(f)
            print('for %s labels shape is :'%madefor,labels.shape)
        
        else: raise ValueError('check your goal_du !')

        while 1:
            if madefor=='train':count=0
            elif madefor=='valid':valid_count=0
            elif madefor=='test':test_count=0
            else:
                raise ValueError('Check your key word')
            for i in range(total_size // bs):  ##i 为该第几个iter，
                    png_input = []
                    while len(png_input)<bs:
                        if madefor=='train':
                            png_input.append(self.concat_many_years(count,concat_years=self.concat_years,madefor=madefor,same_day=self.same_day))
                            count+=1

                        elif madefor=='valid':
                            png_input.append(self.concat_many_years(valid_count,concat_years=self.concat_years,madefor=madefor,same_day=self.same_day))
                            valid_count+=1  

                        elif madefor=='test':
                            png_input.append(self.concat_many_years(test_count,concat_years=self.concat_years,madefor=madefor,same_day=self.same_day))
                            test_count+=1
                            
                        else:
                            raise ValueError(' Check your key word')
                    png_input=np.array(png_input,dtype=np.float32)
                    
                    # if madefor=='train':print('count %s'%count)
                    yield png_input, eqs_datas[i * bs:(i + 1) * bs],labels[i * bs:(i + 1) * bs].reshape([-1,self.num_patches,1])
    
    def generate_arrays_from_file_extendedfeature(self,madefor,bs,concat_years):
        global count
        rootdir='/project/2021eqs_predict/my_dataset/'
        datasetdir=''

        if self.goal_du==2:goal_dustr='2du_'
        else:goal_dustr=''
        
        if self.extendedXL_feature:extendedfeature_str='XL'
        else:extendedfeature_str=''
        datasetdir+='%sextended%s_'%(goal_dustr,extendedfeature_str)
        if self.newdots:datasetdir+='map_v2_'
        datasetdir+='eqs_data_and_labels_with_Mainland'
        if self.scalermode=='maxmin':datasetdir+='_maxmin'
        print(datasetdir)

        # labels_path=os.path.join(rootdir,'%slabels_for_model'%goal_dustr,'%slabels_mag_C_class'%goal_dustr)
        data_path=os.path.join(rootdir,datasetdir,'%seqs_and_png_data_for_'%goal_dustr)
        month_day_list_path=os.path.join(rootdir,datasetdir,'month_day_list.pickle')
       
        # labels=labels['%sd_C%s'%(self.wsize,self.num_classes)].values
        with open('/tic_disk/huyumeng/project/2021eqs_predict/my_dataset/labels_for_model/labels_all_C%s.pickle'%(self.num_classes), 'rb') as f:
            labels=pickle.load(f)
        labels=labels[['%sd'%self.wsize,'%sd_C%s'%(self.wsize,self.num_classes),'date']]
        
        with open(month_day_list_path,'rb') as f: 
            month_day_list=pickle.load(f)
        print('month_day_list:',month_day_list)
        with open(month_day_list_path,'rb') as f: 
            month_day_list=pickle.load(f)
        while 1:
            random.shuffle(month_day_list)
            # print('month_day_list:',month_day_list)
            # raise ValueError
            for i in month_day_list:
                # print('month_day:',i)
                with open(data_path+'%s_%sy_in_%s.pickle'%(madefor,concat_years,i), 'rb') as f:
                    data=pickle.load(f)   
                png_data=data['png']
                eqs_data=data['eq_data']
                years=list(range(len(eqs_data)))[-self.finetuneyears:]
                random.shuffle(years)
                # print('years list:',years)
                
                
                # raise ValueError
                count=0
                if self.selfunknown:selfunknown_num=1
                else:selfunknown_num=0
                for iter in range(len(years)//bs-selfunknown_num):
                    png_input = []
                    eqs_input = []
                    label_input = []
                    while len(png_input)<bs:
                        if not self.selfunknown and madefor=='train':
                            year=years[count]
                            nowdate=date((self.concat_years-1+year+1971),int(i.split('_')[0]),int(i.split('_')[1]))
                            # print(nowdate)
                            todaylabels=labels[labels['date']==nowdate]
                            # print('todaylabels:',todaylabels)
                            png_input.append(self.concat_many_years_same_day(png_data,count=year,concat_years=self.concat_years,madefor=madefor))
                            eqs_input.append(eqs_data[year])
                            label_input.append(todaylabels['%sd_C%s'%(self.wsize,self.num_classes)].values)
                            # print(len(label_input))
                            count+=1
                        elif self.selfunknown and madefor=='train':
                            
                            year=years[count]
                            if year==0 : pass
                            else:
                                png_input.append(self.concat_many_years_same_day(png_data,count=year,concat_years=self.concat_years,madefor=madefor))
                                eqs_input.append(eqs_data[year])
                                # label_input.append(labels[year-1])    #对新label没改这个，不能用
                            count+=1
                            
                        else:
                            raise ValueError(' Check your key word')
                    png_input=np.array(png_input,dtype=np.float32)
                    eqs_input=np.array(eqs_input,dtype=np.float32)
                    label_input=np.array(label_input,dtype=np.float32).reshape([-1,self.num_patches-1,1])
                   
                    yield png_input, eqs_input,label_input


    def get_batch_data(self,madefor,extended=False):
        if madefor in ['train','valid','test']:
            begin_date,end_date=self.get_begin_and_end_date_for_split_dataset(madefor=madefor,concat_years=self.concat_years)
            total_size=(end_date-begin_date).days
            if extended or self.extendedXL_feature:
                GEN=self.generate_arrays_from_file_extendedfeature(bs=self.batch_size,concat_years=self.concat_years,madefor=madefor)
            else:
                GEN=self.generate_arrays_from_file(total_size=total_size,bs=self.batch_size,concat_years=self.concat_years,madefor=madefor)
            return total_size,GEN
        else:
            raise ValueError('Check your key word!!!')
    
    def save_model(self,epoch,global_step,feed_dict):
            '''
            goal 为model——dir下面的模型目标，如这一批是在尝试lr则goal=‘try_lr/lr_0.1’
            '''

            acc_,f1_w,f1_m=self.sess.run([self.accuracy_op,self.f1_op,self.f1_op_macro],feed_dict=feed_dict)
            if os.path.exists('/data/model_dir/my_model/%s/'%self.goal):pass
            else:os.makedirs('/data/model_dir//my_model/%s/'%self.goal)

            self.saver.save(self.sess,
                            '/data/model_dir//my_model/%s//epoch_%02d_lr_%s.ckpt' % (self.goal,epoch, self.learning_rate,))
            print('model saved !')
    def get_level(self,magnitude, maglst,withequal=False):
        """
        给定等级和级别分段，求属于哪个段
        param:
            magnitude: 震级
            maglst: 级别分段，如[3.5, 5, 7]
            withequal:是否取大于等于。
        return:
            int: 级别段， 如f(4, [3.5, 5, 7])=0, f(5.1, [3.5, 5, 7])=1, f(8, [3.5, 5, 7])=2
        """
        maglevel = -1
        for i in range(len(maglst)):
            if withequal:                   ###取大于等于
                if magnitude >=maglst[i]:
                    maglevel = i
            else:
                if magnitude > maglst[i]:
                    maglevel = i
        return maglevel

    def get_different_classify(self,num_class):
        if num_class == 6:
            maglist = [5,5.5,6,6.5,7]
        elif num_class == 4:
            maglist = [5,6,7]
        elif num_class == 2:
            maglist = [5]
        else:
            raise ValueError('Check your num_class!')
        return maglist 

    def get_patch_order(self,patchname):
        if self.goal_du==4:
            png_list=pd.read_csv('/tic_disk/huyumeng/project/2021eqs_predict/png_list_to_patchxy.csv')

        elif self.goal_du==2: 
            png_list=pd.read_csv('/tic_disk/huyumeng/project/2021eqs_predict/png_list_to_patchxy_2du.csv')
        
        else: raise ValueError('check your goal_du!')
        data_order=png_list.set_index('xy').iloc[:,0].to_dict()
        print(data_order)
        patch_order=data_order[patchname]
        print('patch_order:',patch_order)
        return patch_order


    def train_model(self):
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        hparam = self.make_hparam_string()
       
        global_step=0
        train_total_size,train_GEN = self.get_batch_data(madefor='train',extended=self.extendedfeature)
        # valid_total_size,valid_GEN = self.get_batch_data(madefor='valid')
        for i in range(self.epochs):
            batch_num=0
            valid_pre=[]
            valid_true=[]
            valid_logits=[]
            print('drop_out is ',self.drop_out)   
            for png_data,eqs_data,labels in tqdm(train_GEN):
                # print('original labels shape:',np.array(labels).shape)
                # print(labels)
                if self.singlepatchname!='':
                    labels=labels[:,self.get_patch_order(self.singlepatchname),:]
                    # print('single patch labels shape:',labels.shape)
                    # print(labels)
                    labels=labels.reshape(-1,1,1)
                    # print('single patch labels shape:',labels.shape)
                    # print('now:\n',labels)
                    raise ValueError
                if self.learning_rate_decay:
                    feed_dict={self.x_png: png_data, self.x_eqs: eqs_data, self.y:labels, self.drop_out_in_graph:self.drop_out,self.g_step:global_step, self.is_train_valid_test:0,self.selfunknown_tf:self.selfunknown }
                    if batch_num==train_total_size//self.batch_size-1:print('now lr is :',self.sess.run(self.learning_rate2,feed_dict=feed_dict))
                else:
                    feed_dict={self.x_png: png_data, self.x_eqs: eqs_data, self.y:labels, self.drop_out_in_graph:self.drop_out, self.is_train_valid_test:0, self.selfunknown_tf:self.selfunknown}
                
                self.sess.run(self.train_step,feed_dict=feed_dict)
              
                if batch_num%1000 == 0:
                    print('----now is epoch%s step %s-----with learning_rate %s -----'%(i,global_step,self.learning_rate))
                    # logits_,pre_,loss_,=(sess.run([logits,pre,loss,] ,feed_dict=feed_dict))
                    self.sess.run(tf.local_variables_initializer())
                    print('look what initial each batch',[str(var.name) for var in tf.local_variables()]) 
                    logits_,pre_,loss_,=(self.sess.run([self.logits,self.pre,self.loss,] ,feed_dict=feed_dict))
            
                    y_true=labels.flatten()
                    y_pre=pre_.flatten()
                    self.Evaluation(y_t=y_true, y_p=y_pre,num_classes=self.num_classes)
                    print('loss:',loss_,)
                    
                
                batch_num+=1
                global_step+=1
                # if batch_num==2:

                if batch_num==train_total_size//self.batch_size:
                    self.save_model(feed_dict=feed_dict,epoch=i,global_step=global_step)
                    break
            print('----------now is epoch%s-----validation--------------'%i)
           
            self.eval_model_simply(save=False)
            
    
    def load_model(self,modelpath):
        self.saver.restore(self.sess, modelpath)

    def one_hot_for_eval(self,labels):
      
        #连续的整型标签值
        labels=np.array(labels,dtype=int).flatten()
        #one_hot编码
        one_hot_codes = np.eye(self.num_classes)

        one_hot_labels = []
        for label in labels:
            # print(label)
            #将连续的整型值映射为one_hot编码
            one_hot_label = one_hot_codes[label]
            one_hot_labels.append(one_hot_label)

        one_hot_labels = np.array(one_hot_labels)
        # print('check onehot and label',one_hot_labels[:5],labels[:5])
        # print('after onehot shape:',one_hot_labels.shape)
        return one_hot_labels
    
    def save_valid_result(self,valid_pre,valid_logits,valid_true,epoch=None):
        if epoch is None:
            result_save_path='/results/eval_results'
        else:
            result_save_path='my_result_dir/%s/lr_%s/bs_%s/epoch_%s'%(self.goal,self.learning_rate,self.batch_size,epoch)

        if os.path.exists(result_save_path):pass
        else:os.makedirs(result_save_path)
        with open(os.path.join(result_save_path,'eval_1pre_2logits_3label.pickle'), 'wb') as f:
            pickle.dump({'pre':valid_pre,'logits':valid_logits,'label':valid_true},f)
        print('result saved !')

        
    def result_dir(self,epoch=None):
        if epoch is None:
            result_save_path='/results/eval_results/'
        else: 
            result_save_path='/results/%s/lr_%s/bs_%s/epoch_%s'%(self.goal,self.learning_rate,self.batch_size,epoch)

        if os.path.exists(result_save_path):pass
        else:os.makedirs(result_save_path)
        return result_save_path
   
    def get_predict_data_extendedfeature(self):
        rootdir='/tic_disk/huyumeng/project/2021eqs_predict/my_dataset/'
        datasetdir=''

        if self.goal_du==2:goal_dustr='2du_'
        elif self.goal_du==4:goal_dustr=''
        else: raise ValueError('Check your goal_du!')
        
        if self.extendedXL_feature:extendedfeature_str='XL'
        else:extendedfeature_str=''

        datasetdir+='%sextended%s_'%(goal_dustr,extendedfeature_str)
        if self.newdots:datasetdir+='map_v2_'
        datasetdir+='eqs_data_and_labels'
        if self.newlunar:datasetdir+='_newlunar'
        data_path=os.path.join(rootdir,datasetdir,'%seqs_and_png_data_for_predict_%sy_in_11_15.pickle'%(goal_dustr,self.concat_years))
       
        with open(data_path,'rb') as f:
            data=pickle.load(f)
       
        png_data = []
        eqs_data=data['eq_data']
        begin_date,end_date=self.get_begin_and_end_date_for_split_dataset(madefor='test',concat_years=self.concat_years)
        print(len(eqs_data),begin_date,end_date)
        for i in range((end_date.year-begin_date.year)+1):
            png_data.append(data['png'][i:i+self.concat_years])
            
        png_input=np.array(png_data,dtype=np.float32)
        png_input=png_input.reshape([png_input.shape[0],png_input.shape[1]*png_input.shape[2],png_input.shape[3],png_input.shape[4],png_input.shape[5]])
        eqs_input=np.array(eqs_data,dtype=np.float32)
        print(eqs_input.shape,png_input.shape)
        return png_input,eqs_input

    def predict(self,epoch=None):
        '''
        对未来的最新一年进行预测，没有标签可供验证。
        '''
        if self.extendedfeature or self.extendedXL_feature:
            pre_png,pre_eqs_data=self.get_predict_data_extendedfeature()
        else:
            raise Valueerror('Check your feature!')
        pre_pre=[]
        pre_logits=[]
        
        feed_dict={self.x_png: pre_png, self.x_eqs: pre_eqs_data, self.drop_out_in_graph:0, self.is_train_valid_test:2,self.selfunknown_tf:self.selfunknown}
        pre_v, y_logits= (self.sess.run([self.pre,self.logits] ,feed_dict=feed_dict))
        pre.append(pre_v)
        pre_logits.append(y_logits)
          
        pre_logits = np.array(pre_logits)
        pre = np.array(pre_pre)
  
        result_save_path = self.result_dir(epoch=epoch)
        with open(os.path.join(result_save_path,'predict_1pre_2logits.pickle'), 'wb') as f:
            pickle.dump({'pre':pre,'logits':test_logits},f)
        print('Prediction saved !')
    

    def get_extended_valid_data(self):

        datasetdir=''
        if self.goal_du==2:goal_dustr='2du_'
        elif self.goal_du==4:goal_dustr=''
        else: raise ValueError('Check your goal_du!')
        
        if self.extendedXL_feature:extendedfeature_str='XL'
        else:extendedfeature_str=''
        
        datasetdir+='%sextended%s_'%(goal_dustr,extendedfeature_str)
        if self.newdots:datasetdir+='map_v2_'
        # datasetdir+='eqs_data_and_labels'
        datasetdir+='eqs_data_and_labels_with_Mainland'
        if self.scalermode=='maxmin':datasetdir+='_maxmin'
        if self.newlunar:datasetdir+='_newlunar'
        if self.goal_du==2:
            labels_path=os.path.join(self.rootdir,'2du_extended_eqs_data_and_labels','%slabels_mag_C_class%s_for_eval_%dy_in_11_16.pickle'%(goal_dustr,self.num_classes,self.concat_years))
        else:
            labels_path=os.path.join(self.rootdir,'labels_for_model','%slabels_mag_C_class%s_for_eval_%dy_in_11_16.pickle'%(goal_dustr,self.num_classes,self.concat_years))
        data_path=os.path.join(self.rootdir,datasetdir,'%seqs_and_png_data_for_eval_%sy_in_11_16.pickle'%(goal_dustr,self.concat_years))
      
        with open(data_path,'rb') as f:
            data=pickle.load(f)
        with open(labels_path,'rb') as f:
            labels=pickle.load(f)
        
        png_data = []
        eqs_data=data['eq_data']
        
        begin_date,end_date=self.get_begin_and_end_date_for_split_dataset(madefor='valid',concat_years=self.concat_years)
        print(len(labels),len(eqs_data),begin_date,end_date)
        print(np.array(data['png']).shape)
        for i in range((end_date.year-begin_date.year)+1):
            png_data.append(data['png'][i:i+self.concat_years])
        
        png_input=np.array(png_data,dtype=np.float32)
        # print('png_input',png_input.shape)
        # raise ValueError
        png_input=png_input.reshape([png_input.shape[0],png_input.shape[1]*png_input.shape[2],png_input.shape[3],png_input.shape[4],png_input.shape[5]])
        png_input=png_input[:,:,:,:,:self.png_channels] 
        eqs_input=np.array(eqs_data,dtype=np.float32)
        label_input=np.array(labels,dtype=np.float32).reshape([-1,self.num_patches-1,1])    #global token 不要
        return png_input,eqs_input,label_input

    def for_auc(self,valid_true,valid_logits):
        valid_logits=np.array(valid_logits).reshape([-1,self.num_classes])
        valid_true=np.array(valid_true).reshape([-1,1])
        valid_true = self.one_hot_for_eval(valid_true)
        # print('roc-auc macro:',roc_auc_score(y_true=valid_true,y_score=valid_logits,average='macro',multi_class='ovo'))
        # print('valid_prob:\n',valid_logits)

    def eval_model_simply(self,save=True,epoch=None):
        if self.extendedfeature:
            valid_png, valid_eqs_data, valid_true = self.get_extended_valid_data()
        else: raise ValueError('Check your feature!')
        
        feed_dict={self.x_png: valid_png, self.x_eqs: valid_eqs_data, self.y:valid_true,self.drop_out_in_graph:0, self.is_train_valid_test:1, self.selfunknown_tf:self.selfunknown}
        self.sess.run(tf.local_variables_initializer())
        valid_pre,valid_logits= (self.sess.run([self.pre,self.prob] ,feed_dict=feed_dict))
        valid_true = self.get_data_transpose(valid_true,num_patches=self.num_patches-1)[:,1:,:]
        valid_pre = self.get_data_transpose(valid_pre,num_patches=self.num_patches-1)[:,1:,:]
        valid_logits = self.get_data_transpose(valid_logits, num_patches=self.num_patches-1)[:,1:,:]
        print(valid_pre.shape,valid_logits.shape,valid_true.shape)
        # print('first year check',valid_logits[:,1,:],valid_true[:,1,:])
        # raise ValueError
        if save:
            self.save_valid_result(valid_pre,valid_logits,valid_true,epoch=epoch)
        y_5prob=np.expand_dims(np.sum(valid_logits[:,:,1:],axis=-1),-1)
        # print('up 5 prob:',y_5prob[:5],y_5prob.shape)
        y_5t=np.array(np.greater_equal(valid_true,1),'int32')
        # y_5proball=np.concatenate([np.expand_dims(valid_logits[:,:,0],-1),y_5prob],axis=-1)
        # print(y_5proball.shape,y_5t.shape)
        # print('\n______________ALL  Patches_____eval_result_____________')
        # print('binary roc-auc :',roc_auc_score(y_true=np.reshape(y_5t,[-1]),y_score=np.reshape(y_5prob,[-1])))
        each_mag_list_main=self.Evaluation(y_t=valid_true, y_p=valid_pre,num_classes=self.num_classes)
        self.for_auc(valid_true=valid_true, valid_logits=valid_logits)
        
        # print('\n____________Nonzero Patches_______eval_result_____________')

        delete_list=[]
        for patch in range(valid_true.shape[0]):
            if valid_true[patch].max()==0:
                delete_list.append(patch)
        valid_true = np.delete(valid_true,delete_list,axis=0)
        valid_pre = np.delete(valid_pre, delete_list, axis=0)
        valid_logits = np.delete(valid_logits, delete_list, axis=0)
        each_mag_list_nonzero=self.Evaluation(y_t=valid_true, y_p=valid_pre,num_classes=self.num_classes,patch_mode='Nonzero')
        self.for_auc(valid_true=valid_true, valid_logits=valid_logits)
        return  {'allpatches':each_mag_list_main,'nonzero':each_mag_list_nonzero}

    def get_data_transpose(self,want_list,num_patches):
        '''
        for eval_model_simply,已经挑选好最终验证的数据，无需再提取，直接transpose成可视化需要的形状即可(num_patches,concat_years,num_classes)
        '''
        want_list=want_list.reshape([-1,num_patches,want_list.shape[-1]])
        want_list=np.transpose(np.array(want_list),(1,0,2))
        # print('after transpose the shape is :', want_list.shape)
        return want_list

    def get_patch_attention(self,epoch=None,firstlayer=False):
        if self.extendedfeature:
            valid_png, valid_eqs_data, valid_true = self.get_extended_valid_data()
        else:
            valid_png, valid_eqs_data, valid_true = self.get_valid_data()
        
        feed_dict={self.x_png: valid_png, self.x_eqs: valid_eqs_data, self.y:valid_true,self.drop_out_in_graph:0,self.is_train_valid_test:1}
        
        result_save_path = self.result_dir(epoch=epoch)
        if firstlayer:
            savename='eval_attention_map_firstlayer.pickle'
            attention_score= (self.sess.run(self.attention_score_list ,feed_dict=feed_dict))
            print(np.array(attention_score).shape)            
            attention_score=attention_score[0]
            print(np.array(attention_score).shape)            

        else:
            savename='eval_attention_map.pickle'
            attention_score= (self.sess.run(self.attention_score ,feed_dict=feed_dict))
        attention_score=np.array(attention_score)
        print(attention_score.shape)
        attention_score=np.array(np.split(attention_score, self.num_heads, axis=0))
        print(attention_score.shape)
        with open(os.path.join(result_save_path,savename), 'wb') as f:
                pickle.dump((attention_score),f)
            
        print('Attention map saved in: %s'%os.path.join(result_save_path,savename))
        return 
        

    

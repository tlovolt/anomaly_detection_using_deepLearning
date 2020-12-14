import keras
import os
import tensorflow as tf
from keras import applications
from keras import backend as K
from keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D, Conv2D
from keras.layers import BatchNormalization, Activation
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


class ResNet_SF:
    def resnet_layer(self,
                     inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):

        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v1(self, input_shape, depth):

        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x,
                                      num_filters=num_filters,
                                      strides=strides)
                y = self.resnet_layer(inputs=y,
                                      num_filters=num_filters,
                                      activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                          num_filters=num_filters,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=None,
                                          batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def resnet_v2(self, input_shape, depth):
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = self.resnet_layer(inputs=inputs,
                              num_filters=num_filters_in,
                              conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2  # downsample

                # bottleneck residual unit
                y = self.resnet_layer(inputs=x,
                                      num_filters=num_filters_in,
                                      kernel_size=1,
                                      strides=strides,
                                      activation=activation,
                                      batch_normalization=batch_normalization,
                                      conv_first=False)
                y = self.resnet_layer(inputs=y,
                                      num_filters=num_filters_in,
                                      conv_first=False)
                y = self.resnet_layer(inputs=y,
                                      num_filters=num_filters_out,
                                      kernel_size=1,
                                      conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                          num_filters=num_filters_out,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=None,
                                          batch_normalization=False)
                x = keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(2, activation='sigmoid', kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def __init__(self, version=1, n=3):
        self.version = version
        if self.version == 1: self.depth = n * 6 + 2
        elif self.version == 2: self.depth = n * 9 + 2

    def __call__(self, input_shape=[224,224,1]):
        if self.version == 1:
            self.model = self.resnet_v1(input_shape=input_shape, depth=self.depth)
        elif self.version == 2:
            self.model = self.resnet_v2(input_shape=input_shape, depth=self.depth)
        return self.model

########### Focal Loss ###########
##################################
#       |     y_true      |      #
#  ------------------------      #
#       |   0    |   1    |      #
#  ------------------------      #
#  pt_1 |   1    | y_pred |      #
#  ------------------------      #
#  pt_0 | y_pred |   0    |      #
#  ------------------------      #
##################################
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0+K.epsilon()))
    return focal_loss_fixed


def focal_loss_w_ls(gamma=2., alpha=.25, beta=0.2):
    def focal_loss_ls(y_true, y_pred):
        #binary classification
        n_classes = 2

        ls_1 = tf.where(tf.equal(y_true, 1), beta/n_classes + 1 - beta, beta/n_classes)
        ls_0 = tf.where(tf.equal(y_true, 0), beta/n_classes + 1 - beta, beta/n_classes)
        f_1 = K.pow(1. - y_pred, gamma) * K.log(K.epsilon() + y_pred) * ls_1
        f_0 = K.pow(y_pred, gamma) * K.log(K.epsilon() + 1 - y_pred) * ls_0
        return -K.mean(alpha * f_1) - K.mean((1 - alpha) * f_0)
    return focal_loss_ls


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



class TL:
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.pretrained_path = os.path.join(paths.root_dir, 'pretrained')
        self.pretrained_model_weights = None

        if self.model_name == 'resnet50':
            self.pretrained_model_weights = \
                os.path.join(self.pretrained_path, 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        elif self.model_name == 'densenet121':
            self.pretrained_model_weights = \
                os.path.join(self.pretrained_path, 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
        elif self.model_name == 'efficientnetb0':
            self.pretrained_model_weights = \
                os.path.join(self.pretrained_path, 'efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')

    def __call__(self, input_shape=[224,224,3]):
        if self.model_name == 'resnet50':
            base_model = applications.ResNet50(weights= None,
                                               include_top=False,
                                               input_shape=input_shape)
            # base_model = applications.ResNet50(weights=self.pretrained_model_weights,
            #                                    include_top=False,
            #                                    input_shape=input_shape)
        elif self.model_name == 'densenet121':
            base_model = applications.DenseNet121(weights=self.pretrained_model_weights,
                                                  include_top=False,
                                                  input_shape=input_shape)
        elif self.model_name == 'efficientnetb0':
            # base_model = efn.EfficientNetB0(weights=self.pretrained_model_weights,
            #                                 include_top=False,
            #                                 input_shape=input_shape)
            base_model = efn.EfficientNetB0(weights=None,
                                            include_top=False,
                                            input_shape=input_shape)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        preds = Dense(2, activation='softmax')(x)
        self.model = Model(input=base_model.input, output=preds)

        return self.model
    # def __call__(self, input_shape=[224,224,3], selected_top='activation_49'):
    #     base_model = applications.ResNet50(weights=self.pretrained_model_weights,
    #                                        include_top=False,
    #                                        input_shape=input_shape)
    #
    #     x = base_model.get_layer(selected_top).output
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(256, activation='relu')(x)
    #     x = Dropout(0.5)(x)
    #     preds = Dense(2, activation='softmax')(x)
    #     self.model = Model(input=base_model.input, output=preds)
    #
    #     return self.model

if __name__== "__main__":
    resnet = ResNet_SF(version=1)()
    resnet.summary()
    resnet.compile(loss=[focal_loss(alpha=.25, gamma=2)],
                   optimizer=Adam(lr=lr_schedule(0)),
                   metrics=['accuracy'])

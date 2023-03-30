import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPool2D,
    Flatten, Dense, Dropout, Activation, Add,
    ZeroPadding2D, AveragePooling2D, Concatenate,
    GlobalAveragePooling2D, SeparableConv2D, DepthwiseConv2D
)
from tensorflow.keras.regularizers import l2

###### ALEXNET50 #######

# AlexNet: input_shape = (227, 227, 3) & output_shape = 10 (softmax)
# As AlexNet is a rather simple Architecture, we use Sequential API for this.


def AlexNet(classes=10, img_shape=(227, 227, 3)):
    return tf.keras.models.Sequential(layers=[
        Input(shape = img_shape),
        Conv2D(filters=96, kernel_size=11, strides=4,
               activation='relu', use_bias=False, input_shape=img_shape),
        BatchNormalization(),
        MaxPool2D(pool_size=3, strides=2),

        Conv2D(filters=256, kernel_size=5, strides=1,
               activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        MaxPool2D(pool_size=3, strides=2),

        Conv2D(filters=384, kernel_size=3, strides=1,
               activation='relu', padding='same', use_bias=False),
        BatchNormalization(),

        Conv2D(filters=384, kernel_size=3, strides=1,
               activation='relu', padding='same', use_bias=False),
        BatchNormalization(),

        Conv2D(filters=256, kernel_size=3, strides=1,
               activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        MaxPool2D(pool_size=3, strides=2),

        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),

        Dense(4096, activation='relu'),
        Dropout(0.5),

        Dense(classes, activation='softmax')
    ], name="AlexNet")


# from now for all complex models lets use Functional API.


###### RESNET50 #######
# first implement identity stack for resnet50
def res_identity(filters):
    def _res_id(x):
        x_skip = x
        f1, f2 = filters

        x = Conv2D(f1, kernel_size=1, strides=1, padding='valid',
                   kernel_regularizer=l2(0.001), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(f1, kernel_size=3, strides=1, padding='same',
                   kernel_regularizer=l2(0.001), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(f2, kernel_size=1, strides=1, padding='valid',
                   kernel_regularizer=l2(0.001), use_bias=False)(x)
        x = BatchNormalization()(x)

        x = Add()([x, x_skip])
        x = Activation("relu")(x)

        return x
    return _res_id


# defining residual conv stack for resnet50
def res_conv(s, filters):
    def _res_conv(x):
        x_skip = x
        f1, f2 = filters

        x = Conv2D(f1, kernel_size=1, strides=s, padding='valid',
                   kernel_regularizer=l2(0.001), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(f1, kernel_size=3, strides=1, padding='same',
                   kernel_regularizer=l2(0.001), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(f2, kernel_size=1, strides=1, padding='valid',
                   kernel_regularizer=l2(0.001), use_bias=False)(x)
        x = BatchNormalization()(x)

        x_skip = Conv2D(f2, kernel_size=1, strides=s, padding='valid',
                        kernel_regularizer=l2(0.001), use_bias=False)(x_skip)
        x_skip = BatchNormalization()(x_skip)

        x = Add()([x, x_skip])
        x = Activation("relu")(x)

        return x
    return _res_conv

# putting all of it together.
def ResNet50(img_shape=(227, 227, 3), classes=10):

    x = inputs = Input(shape=img_shape)
    x = ZeroPadding2D(padding=3)(x)

    x = Conv2D(64, kernel_size=7, strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = res_conv(s=1, filters=(64, 256))(x)
    for _ in range(2):
        x = res_identity(filters=(64, 256))(x)

    x = res_conv(s=2, filters=(128, 512))(x)
    for _ in range(3):
        x = res_identity(filters=(128, 512))(x)

    x = res_conv(s=2, filters=(256, 1024))(x)
    for _ in range(5):
        x = res_identity(filters=(256, 1024))(x)

    x = res_conv(s=2, filters=(512, 2048))(x)
    for _ in range(2):
        x = res_identity(filters=(512, 2048))(x)

    x = AveragePooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    outputs = Dense(classes, activation='softmax',
                    kernel_initializer='he_normal')(x)

    return tf.keras.Model(inputs, outputs, name='Resnet50')


###### GoogLeNet / Inception V1 ######

def Inception_block(f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
    def _inception_blk(x):
        # 1st path:
        path1 = Conv2D(filters=f1, kernel_size=1,
                       padding='same', activation='relu')(x)

        # 2nd path
        path2 = Conv2D(filters=f2_conv1, kernel_size=1,
                       padding='same', activation='relu')(x)
        path2 = Conv2D(filters=f2_conv3, kernel_size=3,
                       padding='same', activation='relu')(path2)

        # 3rd path
        path3 = Conv2D(filters=f3_conv1, kernel_size=1,
                       padding='same', activation='relu')(x)
        path3 = Conv2D(filters=f3_conv5, kernel_size=5,
                       padding='same', activation='relu')(path3)

        # 4th path
        path4 = MaxPool2D(3, strides=1, padding='same')(x)
        path4 = Conv2D(filters=f4, kernel_size=1,
                       padding='same', activation='relu')(path4)

        output_layer = Concatenate()([path1, path2, path3, path4])

        return output_layer
    return _inception_blk


def GoogLeNet(img_shape=(224, 224, 3), classes=10):
    X = inputs = Input(shape=img_shape)

    X = Conv2D(filters=64, kernel_size=7, strides=2,
               padding='valid', activation='relu')(X)

    X = MaxPool2D(pool_size=3, strides=2)(X)

    X = Conv2D(filters=64, kernel_size=1, strides=1,
               padding='same', activation='relu')(X)

    X = Conv2D(filters=192, kernel_size=3,
               padding='same', activation='relu')(X)

    X = MaxPool2D(pool_size=3, strides=2)(X)

    X = Inception_block(f1=64, f2_conv1=96, f2_conv3=128,
                        f3_conv1=16, f3_conv5=32, f4=32)(X)

    X = Inception_block(f1=128, f2_conv1=128, f2_conv3=192,
                        f3_conv1=32, f3_conv5=96, f4=64)(X)

    X = MaxPool2D(pool_size=(3, 3), strides=2)(X)

    X = Inception_block(f1=192, f2_conv1=96, f2_conv3=208,
                        f3_conv1=16, f3_conv5=48, f4=64)(X)

    # Extra network 1:
    X1 = AveragePooling2D(pool_size=5, strides=3)(X)
    X1 = Conv2D(filters=128, kernel_size=(1, 1),
                padding='same', activation='relu')(X1)
    X1 = Flatten()(X1)
    X1 = Dense(1024, activation='relu')(X1)
    X1 = Dropout(0.7)(X1)
    X1 = Dense(5, activation='softmax')(X1)

    X = Inception_block(f1=160, f2_conv1=112, f2_conv3=224,
                        f3_conv1=24, f3_conv5=64, f4=64)(X)

    X = Inception_block(f1=128, f2_conv1=128, f2_conv3=256,
                        f3_conv1=24, f3_conv5=64, f4=64)(X)

    X = Inception_block(f1=112, f2_conv1=144, f2_conv3=288,
                        f3_conv1=32, f3_conv5=64, f4=64)(X)

    # Extra network 2:
    X2 = AveragePooling2D(pool_size=5, strides=3)(X)
    X2 = Conv2D(filters=128, kernel_size=1,
                padding='same', activation='relu')(X2)
    X2 = Flatten()(X2)
    X2 = Dense(1024, activation='relu')(X2)
    X2 = Dropout(0.7)(X2)
    X2 = Dense(classes, activation='softmax')(X2)

    X = Inception_block(f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32,
                        f3_conv5=128, f4=128)(X)

    X = MaxPool2D(pool_size=3, strides=2)(X)

    X = Inception_block(f1=256, f2_conv1=160, f2_conv3=320,
                        f3_conv1=32, f3_conv5=128, f4=128)(X)

    X = Inception_block(f1=384, f2_conv1=192, f2_conv3=384,
                        f3_conv1=48, f3_conv5=128, f4=128)(X)

    X = GlobalAveragePooling2D()(X)

    X = Dropout(0.4)(X)

    X = Dense(classes, activation='softmax')(X)

    return tf.keras.Model(inputs, [X, X1, X2], name='GoogLeNet')


###### Xception Model #####

def entry_flow(x):

    x = Conv2D(filters=32, strides=2,
               kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64, kernel_size=3,
               strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x_skip = x = Activation('relu')(x)

    x = SeparableConv2D(filters=128, kernel_size=3, strides=1,
                        use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=128, kernel_size=3, strides=1,
                        use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x_skip = Conv2D(filters=128, kernel_size=1, strides=2,
                    use_bias=False, padding='same')(x_skip)
    x_skip = BatchNormalization()(x_skip)
    x = Add()([x, x_skip])

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=256, kernel_size=3, strides=1,
                        use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=256, kernel_size=3, strides=1,
                        use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    x_skip = Conv2D(filters=256, kernel_size=1, strides=2,
                    use_bias=False, padding='same')(x_skip)
    x_skip = BatchNormalization()(x_skip)
    x = Add()([x, x_skip])

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728, kernel_size=3, strides=1,
                        use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728, kernel_size=3, strides=1,
                        use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    x_skip = Conv2D(filters=728, kernel_size=1, strides=2,
                    use_bias=False, padding='same')(x_skip)
    x_skip = BatchNormalization()(x_skip)
    x = Add()([x, x_skip])

    return x


def middle_flow(x):
    x_skip = x
    for __ in range(3):
        x = Activation('relu')(x)
        x = SeparableConv2D(filters=728, kernel_size=3,
                            strides=1, use_bias=False, padding='same')(x)
        x = BatchNormalization()(x)
    return Add()([x, x_skip])


def exit_flow(x, classes):

    x_skip = x
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728, kernel_size=3,
                        strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=1024, kernel_size=3,
                        strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x_skip = Conv2D(filters=1024, kernel_size=1,
                    strides=2, padding='same', use_bias=False)(x_skip)
    x_skip = BatchNormalization()(x_skip)
    x = Add()([x, x_skip])
    x = SeparableConv2D(filters=1536, kernel_size=3,
                        strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=2048, kernel_size=3,
                        strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=classes, activation='softmax')(x)
    return x


def Xception(classes=10, img_shape=(299, 299, 3)):
    x = inputs = Input(shape=img_shape)
    x = entry_flow(x)
    for _ in range(8):
        x = middle_flow(x)
    outputs = exit_flow(x, classes)
    return tf.keras.Model(inputs, outputs, name="Xception")


###### MobileNet v2 ######

def bottleneck(filters, kernel, t, s, res):
    def _bottle_neck(x):
        x_skip = x
        x = Conv2D(filters=x.shape[-1]*t, use_bias=False, strides=1,
                   kernel_size=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU(6)(x)

        x = DepthwiseConv2D(kernel, strides=s, padding="same")(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters, kernel_size=1, strides=1,
                   padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        if res:
            x = Add()([x, x_skip])
        return x
    return _bottle_neck


def inv_res_block(filters, s, n, t=6, kernel=3):
    def _inv_res(x):
        x = bottleneck(filters, kernel, t, s, res=False)(x)
        for _ in range(1, n):
            x = bottleneck(filters, kernel, t, s=1, res=True)(x)
        return x
    return _inv_res


def MobileNet_v2(img_shape=(224, 224, 3), classes=10):
    x = inputs = Input(shape=img_shape)

    x = Conv2D(filters=32, kernel_size=3,
               strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = inv_res_block(filters=16, t=1, s=1, n=1)(x)
    x = inv_res_block(filters=24, s=2, n=2)(x)
    x = inv_res_block(filters=32, s=2, n=3)(x)
    x = inv_res_block(filters=64, s=2, n=4)(x)
    x = inv_res_block(filters=96, s=1, n=3)(x)
    x = inv_res_block(filters=160, s=2, n=3)(x)
    x = inv_res_block(filters=320, s=1, n=1)(x)

    x = Conv2D(filters=1280, kernel_size=(1, 1),
               strides=(1, 1), use_bias=False)(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D(keepdims=True)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(classes, (1, 1), padding='same')(x)
    x = Activation('softmax')(x)
    outputs = Flatten()(x)
    return tf.keras.models.Model(inputs, outputs, name="MobieNetV2")

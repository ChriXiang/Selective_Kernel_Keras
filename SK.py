
def SK(inputs, channel, code, ratio=8):

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    conv1 =  Conv2D(channel, (5, 5), padding='same', name='image_conv_'+code+"_1")(inputs)
    
    conv2 = Conv2D(channel, (3, 3), padding='same', name='image_conv_'+code+"_3")(inputs)
    
    conv_unite = Add()([conv1,conv2])
    
    conv_unite = Activation('relu')(conv_unite)
    
    
    
    avg_pool = GlobalAveragePooling2D()(conv_unite)
    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(conv_unite)
    
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    
    embedding = Add()([max_pool,avg_pool])
    embedding = Activation('softmax')(embedding)
    
    res = K.ones(embedding._keras_shape)
    res = Subtract()([res,embedding])
    
    if K.image_data_format() == "channels_first":
        embedding = Permute((3, 1, 2))(embedding)
        res = Permute((3, 1, 2))(res)
    
    conv1 = multiply([conv1, embedding])
    
    conv2 = multiply([conv2, res])
    
    ans = Add()([conv1,conv2])
    
    return ans

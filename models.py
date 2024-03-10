import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Activation, Concatenate, Add, Dropout
from keras import backend as K


class GatherNd(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GatherNd, self).__init__(**kwargs)
    
    def call(self, image, gather_indices):
        return tf.gather_nd(image, gather_indices)

class ScatterNd(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScatterNd, self).__init__(**kwargs)
    
    def call(self, voxel, scatter_indices, pixels):
        return tf.tensor_scatter_nd_update(voxel, scatter_indices, pixels)

class FastRayTransform(tf.keras.layers.Layer):
    def __init__(self, cameras_LUTs, voxel_size = (200,200,10), **kwargs):
        super(FastRayTransform, self).__init__(**kwargs)
        self.cameras_LUTs = cameras_LUTs # list of LUTs, LUT.shape = (image, gather-scatter, index)
        self.voxel_size = voxel_size #(x,y,z)
        self.gather_nd = GatherNd(dtype='float32')
        self.tensor_scatter_nd_update = ScatterNd(dtype='float32')

    def get_config(self):
        return {
            'cameras_LUTs': self.cameras_LUTs,
            'voxel_size':self.voxel_size,
        }

    @tf.function
    def repeat_for_batch_size(self, indices, batch_size):
        #[[0,0], [1,0]] - > [[0,0,0],[0,1,0],[1,0,0],[1,1,0],[2,0,0],[2,1,0],[3,0,0],[3,1,0]] where first dimension is batch index
        indices_shape = tf.shape(indices)

        batch_range = tf.reshape(tf.range(batch_size, dtype=tf.int32), (batch_size,1))
        #batch_range = tf.repeat(batch_range, (indices.shape[0]), axis=0)#wont work #error None values not supported
        batch_range = tf.repeat(batch_range, (indices_shape[0]), axis=0)
        indices = tf.tile(indices, (batch_size, 1))
        indices= tf.concat([batch_range, indices], axis=1)
        return indices
    
    @tf.function
    def fast_ray_transform(self, LUTs, image, voxel):
        # LUT.shape = (a, index) a = gather or scatter indices
        
        batch_size = tf.shape(image)[0]
        #batch_size = image.shape[0]#wont work #error None values not supported
        
        gather_indices = LUTs[0]
        scatter_indices = LUTs[1]

        gather_indices = self.repeat_for_batch_size(gather_indices, batch_size)
        pixels = self.gather_nd(image, gather_indices)
        scatter_indices = self.repeat_for_batch_size(scatter_indices, batch_size)
        voxel = self.tensor_scatter_nd_update(voxel, scatter_indices, pixels)


        return voxel

    @tf.function
    def call(self, inputs):
        #inpus = list of images, image.shape = (batch, height, width, channels)
        #create voxel, voxel.shape = (batch, x, y, z, channels)
        
        inputs_shape = tf.shape(inputs[0])
        voxel_shape = (inputs_shape[0], *self.voxel_size, inputs_shape[-1])
        #voxel_shape = inputs[0].shape[0:1] + self.voxel_size + inputs[0].shape[-1:] # will not work 
        voxel = tf.zeros(voxel_shape, dtype=inputs[0].dtype)
       
        voxel = self.fast_ray_transform(self.cameras_LUTs[0], inputs[0], voxel)#"CAM_FRONT"
        voxel = self.fast_ray_transform(self.cameras_LUTs[1], inputs[1], voxel)#"CAM_FRONT_LEFT"
        voxel = self.fast_ray_transform(self.cameras_LUTs[2], inputs[2], voxel)#"CAM_FRONT_RIGHT"
        voxel = self.fast_ray_transform(self.cameras_LUTs[3], inputs[3], voxel)#"CAM_BACK"
        voxel = self.fast_ray_transform(self.cameras_LUTs[4], inputs[4], voxel)#"CAM_BACK_LEFT"
        voxel = self.fast_ray_transform(self.cameras_LUTs[5], inputs[5], voxel)#"CAM_BACK_RIGHT"

        return voxel

def conv_block(input, num_filters, normalization):
    x = Conv2D(num_filters, (3,3), padding="same")(input)
    x = NormLayer(x, normalization)(x) 
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3,3), padding="same")(x)
    x = NormLayer(x, normalization)(x)
    x = Dropout(0.2)(x)   
    x = Add()([input, x])
    x = Activation("relu")(x)
    return x

def encoder_block(input, num_filters, normalization):
    x = Conv2D(num_filters, (3,3), padding="same", strides = (2,2))(input)
    x = NormLayer(x, normalization)(x)
    x = Dropout(0.2)(x)  
    x = Activation("relu")(x)
    x = conv_block(x, num_filters, normalization)
    return x 

def decoder_block(input, skip_features, num_filters, normalization):
    x = UpSampling2D(size=(2, 2), interpolation = 'bilinear')(input)
    x = Concatenate()([x, skip_features])
    x = Conv2D(num_filters, (3,3), padding="same")(x)
    x = NormLayer(x, normalization)(x)
    x = Dropout(0.2)(x)   
    x = Activation("relu")(x)
    x = conv_block(x, num_filters, normalization)
    return x

def NormLayer(x, normalization):
    if normalization == 'LayerNormalization':
        layer = tf.keras.layers.LayerNormalization()    
    elif normalization == 'GroupNormalization':
        channels_per_group = 8
        groups = int(x.shape[-1]/channels_per_group)
        layer = tf.keras.layers.GroupNormalization(groups)
    elif normalization == 'BatchNormalization':
        layer = tf.keras.layers.BatchNormalization()
    else:
        raise Exception("a valid normalization method must be provided")
    return layer 

def bev_encoder(input, normalization):
    x = Conv2D(64, (3,3), padding="same")(input)
    x = NormLayer(x, normalization)(x)
    x = Dropout(0.2)(x)  
    x = Activation("relu")(x)
    s1 = conv_block(x, 64, normalization)#200x200
    s2 = encoder_block(s1,128, normalization)#100x100
    s2 = conv_block(s2, 128, normalization)#100x100
    s3 = encoder_block(s2,256, normalization)#50x50
    s3 = conv_block(s3, 256, normalization)#50x50
    s4 = encoder_block(s3,512, normalization)#25x25
    sp = conv_block(s4, 512, normalization)#25x25
    sp2 = conv_block(sp, 512, normalization)#25x25
    sp3 = conv_block(sp2, 512, normalization)#25x25
    p4 = decoder_block(sp3, s3, 256, normalization)#50x50
    p3 = decoder_block(p4, s2, 128, normalization)#100x100
    p2 = decoder_block(p3, s1, 64, normalization)#200x200
    return p2 

def BackBone(input_shape, back_bone='EfficientNetB1', normalization = 'GroupNormalization', weights = 'imagenet'):
    if back_bone == 'EfficientNetB0': model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights = weights)
    if back_bone == 'EfficientNetB1': model = tf.keras.applications.EfficientNetB1(include_top=False, input_shape=input_shape, weights = weights)
    if back_bone == 'EfficientNetB2': model = tf.keras.applications.EfficientNetB2(include_top=False, input_shape=input_shape, weights = weights)
    if back_bone == 'EfficientNetB3': model = tf.keras.applications.EfficientNetB3(include_top=False, input_shape=input_shape, weights = weights)
    if back_bone == 'EfficientNetB4': model = tf.keras.applications.EfficientNetB4(include_top=False, input_shape=input_shape, weights = weights)
    if normalization == 'BatchNormalization': return model

    model_layers_dict = {}
    x = model.layers[0].output
    model_layers_dict[model.layers[0].name] = x
    for layer in model.layers[1:]:
        layer_name = layer.name
        if isinstance(layer.input, list):
            x = [model_layers_dict[ly.name.split("/")[0]] for ly in layer.input]
        else:
            x = model_layers_dict[layer.input.name.split("/")[0]]

        if layer.__class__.__name__ == 'BatchNormalization':
            new_layer = NormLayer(x, normalization)
            x = new_layer(x)
        else:
            layer_config ={'class_name': layer.__class__.__name__, 'config':layer.get_config()}
            new_layer = tf.keras.layers.deserialize(layer_config)
            x = new_layer(x)
            new_layer.set_weights(layer.get_weights())

        model_layers_dict[layer_name] = x
    model = tf.keras.models.Model(model.layers[0].input, x)
    return model

def image_encoder(input_shape, back_bone='EfficientNetB1', normalization = 'GroupNormalization', weights = 'imagenet'):
    model = BackBone(input_shape, back_bone, normalization, weights)
    model  = Model(model.input, model.layers[-4].output)
    blocks = []
    blocks_to_stop = ['block4', 'block6'] 
    block_index = 0
    block_to_stop = blocks_to_stop[block_index]
    for index, layer in enumerate(model.layers):
        if block_to_stop in layer.name:
            blocks.append(model.layers[index-1].output)
            block_index+=1
            if block_index > len(blocks_to_stop)-1: break
            block_to_stop = blocks_to_stop[block_index]
    blocks.append(model.layers[-1].output) 

    x = decoder_block(blocks[-1], blocks[-2], 256, normalization)
    x = decoder_block(x, blocks[-3], 128, normalization)
    x = Conv2D(3, (1, 1), activation='softmax', dtype="float32")(x)
    model = Model(model.input, x)

    return model


def ultra_fast_bev(input_shape, cameras_LUTs, n_classes = 3, back_bone='EfficientNetB1', normalization = 'GroupNormalization', image_encoder_weights_path='./model_weights/image_encoder_effnetb1_groupnorm_bs20_imagenet.h5', modality = 'CL'):
    img_encoder = image_encoder(input_shape, back_bone=back_bone, normalization = normalization, weights = 'imagenet')
    img_encoder.load_weights(image_encoder_weights_path)

    blocks = []
    blocks_to_stop = ['block2','block3','block4'] 
    block_index = 0
    block_to_stop = blocks_to_stop[block_index]
    for index, layer in enumerate(img_encoder.layers):
        if block_to_stop in layer.name:
            blocks.append(img_encoder.layers[index-1].output)
            block_index+=1
            if block_index > len(blocks_to_stop)-1: break
            block_to_stop = blocks_to_stop[block_index]

    img_encoder = tf.keras.models.Model(img_encoder.input, [img_encoder.layers[-2].output, blocks[-1], blocks[-2], blocks[-3]])
    
    inputs = [Input(input_shape) for i in range(6)]
    inputs.append(Input((200,200,4)))#Lidar data

    encoder_input = tf.concat(inputs[0:6], axis = 0) #axis = 0 is the batch axis, 
    img_encoder_output1, img_encoder_output2, img_encoder_output3, img_encoder_output4 = img_encoder(encoder_input)
    
    img_encoder_outputs1 = tf.split(img_encoder_output1, num_or_size_splits=6, axis=0)#img_encoder_output1 shape (6*batch,56,100,128) -> (6,batch,56,100,128)
    img_encoder_outputs2 = tf.split(img_encoder_output2, num_or_size_splits=6, axis=0)#img_encoder_output1 shape (6*batch,56,100,40) -> (6,batch,56,100,40)
    img_encoder_outputs3 = tf.split(img_encoder_output3, num_or_size_splits=6, axis=0)#(6,batch,112,200,24)
    img_encoder_outputs4 = tf.split(img_encoder_output4, num_or_size_splits=6, axis=0)#(6,batch,224,400,16)


    voxel1 = FastRayTransform(cameras_LUTs[0], voxel_size = (200,200,4))(img_encoder_outputs1)#voxel_shape (batch, 200, 200, 4, 128)
    voxel2 = FastRayTransform(cameras_LUTs[1], voxel_size = (200,200,4))(img_encoder_outputs2)#voxel_shape (batch, 200, 200, 4, 40)
    voxel3 = FastRayTransform(cameras_LUTs[2], voxel_size = (200,200,4))(img_encoder_outputs3)#voxel_shape (batch, 200, 200, 4, 24)
    voxel4 = FastRayTransform(cameras_LUTs[3], voxel_size = (200,200,4))(img_encoder_outputs4)#voxel_shape (batch, 200, 200, 4, 16)

    if modality == 'CL':
        #lidar shape(batch,x,y,z) -> shape(batch,x,y,z,1)
        lidar = tf.expand_dims(inputs[6], axis=-1)
        voxel  = tf.concat([voxel1, voxel2, voxel3, voxel4, lidar], axis=-1)#voxel_shape (batch, 200, 200, 4, 128+40+24+16+1)
    elif modality == 'C':
        voxel  = tf.concat([voxel1, voxel2, voxel3, voxel4], axis=-1)

    voxel_shape = voxel.shape
    voxel_shape_simbolic = tf.shape(voxel)

    #voxel_shape shape (batch, x, y, z, channels) -> (batch, x, y, z*channels)
    #voxel = tf.reshape(voxel, voxel_shape[0:3]+(voxel_shape[3]*voxel_shape[4],))# wont work will return shape of (none, none, none) insted of (none, 200, 200, 4*(128+40+24+16+1)) must use simbolic shape 
    voxel = tf.reshape(voxel, (*voxel_shape_simbolic[0:3], voxel_shape_simbolic[3]*voxel_shape_simbolic[4]))  
    voxel = Conv2D(int(voxel_shape[4]/8)*8, (3,3), padding="same")(voxel)#channels must be multiple of 8 so they can be equaly divided into groups 
    voxel = NormLayer(voxel, normalization)(voxel)
    voxel = Dropout(0.2)(voxel)
    voxel = Activation("relu")(voxel)
    voxel = bev_encoder(voxel, normalization)

    out = Conv2D(n_classes, (1, 1), activation='softmax', dtype="float32")(voxel)

    return Model(inputs=inputs, outputs=out)



def ultra_fast_bev_nmsbf(input_shape, cameras_LUTs, n_classes = 3, back_bone='EfficientNetB1', normalization = 'GroupNormalization', image_encoder_weights_path='./model_weights/image_encoder_effnetb1_groupnorm_bs20_imagenet.h5', modality = 'CL'):
    #nmsf = no multy scale bev features 
    img_encoder = image_encoder(input_shape, back_bone=back_bone, normalization = normalization, weights = 'imagenet')
    img_encoder.load_weights(image_encoder_weights_path)

    img_encoder = tf.keras.models.Model(img_encoder.input, img_encoder.layers[-2].output)
    
    inputs = [Input(input_shape) for i in range(6)]
    inputs.append(Input((200,200,4)))#Lidar data

    encoder_input = tf.concat(inputs[0:6], axis = 0) #axis = 0 is the batch axis, 
    img_encoder_output1 = img_encoder(encoder_input)
    
    img_encoder_outputs1 = tf.split(img_encoder_output1, num_or_size_splits=6, axis=0)#img_encoder_output1 shape (6*batch,56,100,128) -> (6,batch,56,100,128)
    voxel1 = FastRayTransform(cameras_LUTs[0], voxel_size = (200,200,4))(img_encoder_outputs1)#voxel_shape (batch, 200, 200, 4, 128)

    if modality == 'CL':
        #lidar shape(batch,x,y,z) -> shape(batch,x,y,z,1)
        lidar = tf.expand_dims(inputs[6], axis=-1)
        voxel  = tf.concat([voxel1, lidar], axis=-1)#voxel_shape (batch, 200, 200, 4, 128+1)
    elif modality == 'C':
        voxel = voxel1
    else:
        raise Exception("modality must be C = Camera or CL = Camera + Lidar")
    voxel_shape = voxel.shape
    voxel_shape_simbolic = tf.shape(voxel)

    #voxel_shape shape (batch, x, y, z, channels) -> (batch, x, y, z*channels)
    #voxel = tf.reshape(voxel, voxel_shape[0:3]+(voxel_shape[3]*voxel_shape[4],))# wont work will return shape of (none, none, none) insted of (none, 200, 200, 10*128) must use simbolic shape 
    voxel = tf.reshape(voxel, (*voxel_shape_simbolic[0:3], voxel_shape_simbolic[3]*voxel_shape_simbolic[4]))  
    voxel = Conv2D(int(voxel_shape[4]/8)*8, (3,3), padding="same")(voxel)#channels must be multiple of 8 so they can be equaly divided into groups 
    voxel = NormLayer(voxel, normalization)(voxel)
    voxel = Dropout(0.2)(voxel)
    voxel = Activation("relu")(voxel)
    voxel = bev_encoder(voxel, normalization)

    out = Conv2D(n_classes, (1, 1), activation='softmax', dtype="float32")(voxel)

    return Model(inputs=inputs, outputs=out)
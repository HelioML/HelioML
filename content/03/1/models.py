from keras.layers import Input, Conv2D, Activation, BatchNormalization, GaussianNoise, add, UpSampling2D
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils

def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: One of `channels_last` or `channels_first`.
    # Returns
        A padded 4D tensor.
    # Raises
        ValueError: if `data_format` is neither
            `channels_last` or `channels_first`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if data_format == 'channels_first':
        pattern = [[0, 0],
                   [0, 0],
                   list(padding[0]),
                   list(padding[1])]
    else:
        pattern = [[0, 0],
                   list(padding[0]), list(padding[1]),
                   [0, 0]]
    return tf.pad(x, pattern, "REFLECT")

class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns or zeros
    at the top, bottom, left and right side of an image tensor.
    # Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to width and height.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, padded_rows, padded_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, padded_rows, padded_cols)`
    """
    
    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    input_shape[1],
                    rows,
                    cols)
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                    padding=self.padding,
                                    data_format=self.data_format)

    def get_config(self):
        config = {'padding': self.padding,
                  'data_format': self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def keepsize(nx, ny, noise, depth, activation='relu', n_filters=64, l2_reg=1e-7):
    """
    Deep residual network that keeps the size of the input throughout the whole network
    """

    def residual(inputs, n_filters):
        x = ReflectionPadding2D()(inputs)
        x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = ReflectionPadding2D()(x)
        x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = add([x, inputs])

        return x

    inputs = Input(shape=(nx, ny, 1))
    x = GaussianNoise(noise)(inputs)

    x = ReflectionPadding2D()(x)
    x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
    x0 = Activation(activation)(x)

    x = residual(x0, n_filters)

    for i in range(depth-1):
        x = residual(x, n_filters)

    x = ReflectionPadding2D()(x)
    x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = add([x, x0])

# Upsampling for superresolution
    x = UpSampling2D()(x)
    x = ReflectionPadding2D()(x)
    x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
    x = Activation(activation)(x)

    final = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)

    return Model(inputs=inputs, outputs=final)


def encdec(nx, ny, noise, depth, activation='relu', n_filters=64):
    """
    Deep residual network using an encoder-decoder approach. It uses reflection padding
    """
    def residual(inputs, n_filters):

        x = ReflectionPadding2D()(inputs)
        x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = ReflectionPadding2D()(x)
        x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)
        x = BatchNormalization()(x)
        x = add([x, inputs])

        return x    

    def residual_down(inputs, n_filters):
        x = ReflectionPadding2D()(inputs)
        x = Conv2D(n_filters, (3, 3), strides=2, padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)        
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        
        x = ReflectionPadding2D()(x)
        x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)        
        x = BatchNormalization()(x)       

        shortcut = Conv2D(n_filters, (1, 1), strides=2, padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(inputs)

        x = add([x, shortcut])

        return x

    def residual_up(inputs, n_filters):
        x_up = UpSampling2D(size=(2,2))(inputs)

        x = ReflectionPadding2D()(x_up)
        x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)        
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

        x = ReflectionPadding2D()(x)                
        x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)
        x = BatchNormalization()(x)        
        
        shortcut = Conv2D(n_filters, (1, 1), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x_up)

        x = add([x, shortcut])

        return x

    inputs = Input(shape=(nx, ny, 1))

# in: (nx,ny,1) -> out: (nx,ny,n_filters)
    x = ReflectionPadding2D()(inputs)
    x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

# in: (nx,ny,n_filters) -> out: (nx/2,ny/2,2*n_filters)
    x = ReflectionPadding2D()(x)
    x = Conv2D(2*n_filters, (3, 3), strides=2, padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

# in: (nx/2,ny/2,2*n_filters) -> out: (nx/4,ny/4,4*n_filters)
    # x = ReflectionPadding2D()(x)
    # x = Conv2D(4*n_filters, (3, 3), strides=2, padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)
    # x = BatchNormalization()(x)
    # x = Activation(activation)(x)

    for i in range(depth):
        x = residual(x, 2*n_filters)

# in: (nx/4,ny/4,4*n_filters) -> out: (nx/2,ny/2,2*n_filters)
    # x = UpSampling2D(size=(2,2))(x)
    # x = ReflectionPadding2D()(x)
    # x = Conv2D(2*n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)        
    # x = BatchNormalization()(x)
    # x = Activation(activation)(x)

# in: (nx/2,ny/2,2*n_filters) -> out: (nx,ny,n_filters)
    x = UpSampling2D(size=(2,2))(x)
    x = ReflectionPadding2D()(x)
    x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)        
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

# in: (nx,ny,n_filters) -> out: (2*nx,2*ny,n_filters)
    x = UpSampling2D(size=(2,2))(x)
    x = ReflectionPadding2D()(x)
    x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)        
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    final = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-7))(x)

    return Model(inputs=inputs, outputs=final)

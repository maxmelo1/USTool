from keras import backend as K
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf2
from tensorflow.keras.losses import binary_crossentropy

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    loss = (1 - jac) * smooth
#     print(loss.shape)
#     print(loss)
    return loss

# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred = K.cast(y_pred, 'float32')
#     y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
#     intersection = y_true_f * y_pred_f
#     score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
#     return score
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    gamma = 2.5
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def  bce_dice_loss(y_true, y_predict): #combination of dice loss and binary cross entropy for all pixels
    return binary_crossentropy(y_true, y_predict) + (1-dice_coef(y_true, y_predict))


# https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ( intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# https://github.com/Golbstein/KerasExtras/blob/master/keras_functions.py
def Mean_IOU_tensorflow_2(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.cast(true_labels & pred_labels, tf.int32) 
        union = tf.cast(true_labels | pred_labels, tf.int32) 
        legal_batches = K.sum(tf.cast(true_labels, tf.int32) , axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(ious[legal_batches]))
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = iou[legal_labels]
    return K.mean(iou)

# def Mean_IOU(y_true, y_pred):
#     nb_classes = K.int_shape(y_pred)[-1]
#     y_pred = K.reshape(y_pred, (-1, nb_classes))
#     y_true = tf.cast(K.reshape(y_true, (-1, 1))[:,0], tf.int32)
#     y_true = K.one_hot(y_true, nb_classes)
#     true_pixels = K.argmax(y_true, axis=-1) # exclude background
#     pred_pixels = K.argmax(y_pred, axis=-1)
#     iou = []
#     flag = tf.convert_to_tensor(-1, dtype='float64')
#     for i in range(nb_classes-1):
#         true_labels = K.equal(true_pixels, i)
#         pred_labels = K.equal(pred_pixels, i)
#         inter = tf.cast(true_labels & pred_labels, tf.int32)
#         union = tf.cast(true_labels | pred_labels, tf.int32)
#         cond = (K.sum(union) > 0) & (K.sum(tf.cast(true_labels, tf.int32)) > 0)
#         res = tf.cond(cond, lambda: K.sum(inter)/K.sum(union), lambda: flag)
#         iou.append(res)
#     iou = tf.stack(iou)
#     legal_labels = tf.greater(iou, flag)
#     iou = tf.gather(iou, indices=tf.where(legal_labels))
#     return K.mean(iou)

def mean_iou(y_true, y_pred):
    num_class = K.int_shape(y_pred)[-1]

    y = tf.argmax(y_true, -1)
    y_hat = tf.argmax(y_pred, -1)

    score, up_opt = tf2.metrics.mean_iou(y, y_hat, num_class)
    tf2.keras.backend.get_session().run(tf2.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

# def mean_iou(y_true, y_pred):
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         y_pred_ = tf.cast(y_pred > t, tf.int32)
#         score, up_opt = tf2.metrics.mean_iou(y_true, y_pred_, 2)
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)


#Keras
def IoULoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

def gen_dice(y_true, y_pred, eps=1e-6):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""

    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    # count how many of each class are present in 
    # each image, if there are zero, then assign
    # them a fixed weight of eps
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / (counts ** 2)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
# def tversky_loss(y_true, y_pred):
#     alpha = 0.3
#     beta  = 0.7
    
#     ones = K.ones(K.shape(y_true))
#     p0 = y_pred      # proba that voxels are class i
#     p1 = ones-y_pred # proba that voxels are not class i
#     g0 = y_true
#     g1 = ones-y_true
    
#     num = K.sum(p0*g0, (0,1,2,3))
#     den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
#     T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
#     Ncl = K.cast(K.shape(y_true)[-1], 'float32')
#     return Ncl-T

def weighted_categorical_crossentropy(weights):
    
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(batch_y_true, batch_y_pred):
        
        batch_y_pred /= tf.reduce_sum(batch_y_pred, -1, True)

        # 先求batch中每个样本的cross entropy值
        batch_y_pred = tf.clip_by_value(batch_y_pred, K.epsilon(), 1)
        batch_loss = - tf.reduce_sum(batch_y_true * tf.math.log(batch_y_pred) * weights, -1)  # 分类时为1， 分割时为3，放-1兼容两者。
        return K.sum(batch_loss)

    return loss

def weighted_categorical_crossentropy2(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


# def categorical_focal_loss(alpha, gamma=2.):
#     """
#     Softmax version of focal loss.
#     When there is a skew between different categories/labels in your data set, you can try to apply this function as a
#     loss.
#            m
#       FL = \sum  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
#           c=1
#       where m = number of classes, c = class and o = observation
#     Parameters:
#       alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
#       categories/labels, the size of the array needs to be consistent with the number of classes.
#       gamma -- focusing parameter for modulating factor (1-p)
#     Default value:
#       gamma -- 2.0 as mentioned in the paper
#       alpha -- 0.25 as mentioned in the paper
#     References:
#         Official paper: https://arxiv.org/pdf/1708.02002.pdf
#         https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
#     Usage:
#      model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """

#     alpha = np.array(alpha, dtype=np.float32)

#     def categorical_focal_loss_fixed(y_true, y_pred):
#         """
#         :param y_true: A tensor of the same shape as `y_pred`
#         :param y_pred: A tensor resulting from a softmax
#         :return: Output tensor.
#         """

#         # Clip the prediction value to prevent NaN's and Inf's
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

#         # Calculate Cross Entropy
#         cross_entropy = -y_true * K.log(y_pred)

#         # Calculate Focal Loss
#         loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

#         # Compute mean loss in mini_batch
#         #return K.mean(K.sum(loss, axis=-1))
#         return K.sum(loss, axis=-1)

#     return categorical_focal_loss_fixed

def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = \sum  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * y_true * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        #return K.mean(K.sum(loss, axis=-1))
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

# def categorical_focal_loss(alpha=None, gamma=2.):
#     """
#     :param alpha: 维度为 [class_no]
#     :param gamma: Float32
#     """

#     def focal_loss(batch_y_true, batch_y_pred):
#         """
#         图片分类：
#         :param batch_y_true: 维度为 [batch_size, class_no]
#         :param batch_y_pred: 维度为 [batch_size, class_no]
#         图片分割：
#         :param batch_y_true: 维度为 [batch_size, image_H, image_W, class_no]
#         :param batch_y_pred: 维度为 [batch_size, image_H, image_W, class_no]
#         """
#         # 归一化，加下面这段是为了兼容最后一层非Softmax的情况，如果是Softmax的输出可以注释掉，因为那个输出已经归一化了。
#         # batch_y_pred /= tf.reduce_sum(batch_y_pred, -1, True)

#         # 防止log(0)为-inf，tf里面 0 * -inf = nan
#         # 虽然也不影响back propagation，它只关心导数，不关心这个loss值。
#         batch_y_pred = tf.clip_by_value(batch_y_pred, K.epsilon(), 1)
#         if alpha:
#             batch_loss = - tf.reduce_sum(alpha * batch_y_true * (1 - batch_y_pred) ** gamma * tf.math.log(batch_y_pred), -1)
#         else:
#             batch_loss = - tf.reduce_sum(batch_y_true * (1 - batch_y_pred) ** gamma * tf.math.log(batch_y_pred), -1)

#         return K.sum(batch_loss)  # 分割情况下用mean,偏导数会非常小,所以这边用sum，当然你用mean也是可以的。

#     return focal_loss


# def mIOU(gt, preds):
#     ulabels = np.unique(gt)
#     iou = np.zeros(len(ulabels))
#     for k, u in enumerate(ulabels):
#         inter = (gt == u) & (preds==u)
#         union = (gt == u) | (preds==u)
#         iou[k] = inter.sum()/union.sum()
#     return np.round(iou.mean(), 2)
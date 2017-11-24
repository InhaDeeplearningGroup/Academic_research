import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow import ConfigProto
slim = tf.contrib.slim

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

import numpy as np
import scipy.io as sio
#4

#train_dir   =  '/home/dmsl3/nas/backup1/personal_lsh/training/cifar100/gram/vgg6_gram'
train_dir   =  '/home/dmsl3/Documents/svd/std_crop'
dataset_dir = '/home/dmsl3/Documents/data/tf/cifar100'

dataset_name = 'cifar100'
model_name   = 'vgg16_std'
preprocessing_name = 'cifar10'

Optimizer = 'sgd' # 'adam' or 'sgd'
Learning_rate =1e-2
learning_rate_decay_factor = 0.9462372

batch_size = 128
val_batch_size = 200
epoch = 200
weight_decay = 1e-4

gpu_num = 2
gpu_id = None
checkpoint_path = None
#checkpoint_path = '/home/dmsl/nas/share/training/cifar100/cifar100'
cpt_scopes = (['name','ADNet'],         ##  name : net's main graph ## conv's param = [batch, gamma]
              ['conv0',[False,True]],# ['conv0/conv1',[True,True]],
              ['conv1',[False,True]],# ['conv1/conv1',[True,True]],
              ['conv2',[False,True]],# ['conv2/conv1',[True,True]],
              )#, ['fc/fc2',[True]])

#checkpoint_path = '/home/dmsl/Documents/tf/cifar10_student'
#cpt_scopes = (['name','cifar10_std'],         ##  name : net's main graph ## conv's param = [batch, gamma]
#              ['conv1',[False,False]],
#              ['conv2',[False,False]],
#              ['conv3',[False,False]],
#              ['conv4',[False,False]],
#              ['conv5',[False,False]],
#              ['conv6',[False,False]])
ignore_missing_vars = True

### main
#%%    
tf.logging.set_verbosity(tf.logging.INFO)
def _get_init_fn(checkpoint_path, cpt_scopes, ignore_missing_vars):
    if checkpoint_path is None:
        return None
    exclusions = []
    if cpt_scopes:
        name = cpt_scopes[0][1]
        for scope, param in cpt_scopes[1:]:
            if scope[:5] == 'batch':
                exclusions.append(name+'/'+scope +'/'+'batch/beta:0')
                if param[0] == True:
                    exclusions.append(name+'/'+scope +'/'+'batch/gamma:0')
                exclusions.append(name+'/'+scope +'/'+'batch/moving_mean:0')
                exclusions.append(name+'/'+scope +'/'+'batch/moving_variance:0')
            elif scope[:4] == 'conv':
                exclusions.append(name+'/'+scope +'/'+ 'weights:0')
                if param[0] == True:
                    exclusions.append(name+'/'+scope +'/'+'batch/beta:0')
                    if param[1] == True:
                        exclusions.append(name+'/'+scope +'/'+'batch/gamma:0')
                    exclusions.append(name+'/'+scope +'/'+'batch/moving_mean:0')
                    exclusions.append(name+'/'+scope +'/'+'batch/moving_variance:0')
                else:
                    exclusions.append(name+'/'+scope +'/'+ 'biases:0')
            elif scope[:5] == 'split':
                exclusions.append(name+'/'+scope +'/'+'kernel:0')
            elif scope[:4] == 'full':
                exclusions.append(name+'/'+scope +'/'+ 'weights:0')
                exclusions.append(name+'/'+scope +'/'+ 'biases:0')
                    
    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
            if not excluded:
                variables_to_restore.append(var)

    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(checkpoint_path,
                                          variables_to_restore,
                                          ignore_missing_vars = ignore_missing_vars)


def GET_dataset(dataset_name, dataset_dir, batch_size, preprocessing_name, split):
    if split == 'train':
        sff = True
        threads = 4
    else:
        sff = False
        threads = 1
    with tf.variable_scope('dataset_%s'%split):
        dataset = dataset_factory.get_dataset(dataset_name, split, dataset_dir)
#        with tf.device('/device:CPU:0'):
        if split == 'train':
            global_step = slim.create_global_step()
            p = tf.floor_div(tf.cast(global_step, tf.float32), tf.cast(int(dataset.num_samples / float(batch_size)), tf.float32))
        else:
            global_step = None
            p = None
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                  shuffle=sff,
                                                                  num_readers = threads,
                                                                  common_queue_capacity=20 * batch_size,
                                                                  common_queue_min=10 * batch_size)
        images, labels = provider.get(['image', 'label'])
            
        if split == 'train':
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name)
            images = image_preprocessing_fn(images)
#            images = tf.to_float(images)
        else:
            images = tf.to_float(images)
            images = (images-np.array([112.4776,124.1058,129.3773]).reshape(1,1,3))/np.array([70.4587,65.4312,68.2094]).reshape(1,1,3)
#            images = tf.concat([(tf.slice(images,[0,0,0],[32,32,1])-112.4776)/70.4587,
#                                (tf.slice(images,[0,0,1],[32,32,1])-124.1058)/65.4312,
#                                (tf.slice(images,[0,0,2],[32,32,1])-129.3773)/68.2094],2)
#            images = tf.squeeze(images)
        if split == 'train':
            batch_images, batch_labels = tf.train.shuffle_batch([images, labels],
                                                    batch_size = batch_size,
                                                    num_threads = threads,
                                                    capacity = 20 * batch_size,
                                                    min_after_dequeue = 10 * batch_size)
        else:
            batch_images, batch_labels = tf.train.batch([images, labels],
                                                         batch_size = batch_size,
                                                         num_threads = threads,
                                                         capacity = 20 * batch_size)
        
        with tf.variable_scope('1-hot_encoding'):
            batch_labels = slim.one_hot_encoding(batch_labels, 100,on_value=1.0)
        batch_queue = slim.prefetch_queue.prefetch_queue([batch_images, batch_labels], capacity=5*batch_size)
        
        image, label = batch_queue.dequeue()
    return p, global_step, dataset, image, label

def MODEL(model_name, weight_decay, image, label, lr, p, is_training, reuse):
    network_fn = nets_factory.get_network_fn(model_name, weight_decay = weight_decay)
    
    end_points = network_fn(image, is_training=is_training, lr = lr, val=reuse)
    
    loss = tf.losses.softmax_cross_entropy(label,end_points['Logits'])
    
    accuracy = slim.metrics.accuracy(tf.to_int32(tf.argmax(end_points['Logits'], 1)),
                                         tf.to_int32(tf.argmax(label, 1)))
    

    if is_training:
        loss2 = end_points['Dist']
        w = 3e-1
        loss2 *= w
        
        loss += loss2
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss2', loss2)
        
        
        return loss, accuracy,0
    
    
    
    return loss, accuracy
#    return total_loss
#%%    
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

with tf.Graph().as_default() as graph:
    ## Load Dataset
    p, global_step, dataset, images, labels = GET_dataset(dataset_name, dataset_dir,
                                                        batch_size, preprocessing_name, 'train')
    _, _, val_dataset, val_image, val_label = GET_dataset(dataset_name, dataset_dir,
                                                          val_batch_size, preprocessing_name, 'test')
    with tf.device('/device:CPU:0'):
        decay_steps = dataset.num_samples // batch_size
        max_number_of_steps = int(dataset.num_samples/batch_size*epoch)
        Learning_rate = tf.cond(tf.greater_equal(p,50),
                                lambda : Learning_rate/10,
                                lambda : Learning_rate)
        Learning_rate = tf.cond(tf.greater_equal(p,100),
                                lambda : Learning_rate/100,
                                lambda : Learning_rate)
        Learning_rate = tf.cond(tf.greater_equal(p,150),
                                lambda : Learning_rate/1000,
                                lambda : Learning_rate)
        
        if Optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(Learning_rate, epsilon = 1e-16)
        elif Optimizer == 'sgd':
            optimizer = tf.train.MomentumOptimizer(Learning_rate, 0.9, use_nesterov=True)
    
    total_loss = []
    gradients = []
    if gpu_id == None:
        gpu_id = list(range(gpu_num))
    for i in range(gpu_num):
        image = tf.slice(images,[i*batch_size//gpu_num,0,0,0],[batch_size//gpu_num,-1,-1,-1])
        label = tf.slice(labels,[i*batch_size//gpu_num,0],[batch_size//gpu_num,-1])
        with tf.device('/gpu:%d' % i):
#            with tf.variable_scope('model_gpu%d' %i):
            if i > 0:
                reuse = True
            else:
                reuse = False
            loss, train_accuracy,loss2 = MODEL(model_name, weight_decay, image, label, Learning_rate, p, True, reuse)
            total_loss.append(loss)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            gradients.append(optimizer.compute_gradients(loss, var_list = variables))
        
    total_loss = tf.add_n(total_loss)
    grads = average_gradients(gradients)
    grad_updates = optimizer.apply_gradients(grads, global_step=global_step)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append(grad_updates)
        
    ## Training
    update_op = tf.group(*update_ops)
    train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')
    
    
    val_loss, val_accuracy = MODEL(model_name, weight_decay, val_image, val_label, Learning_rate, p, False, True)
    
    ## Get Summaries
    for variable in slim.get_model_variables():
        tf.summary.histogram(variable.op.name, variable)
        
    tf.summary.scalar('learning_rate', Learning_rate)    
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')    
    
    def ts_fn(session, *args, **kwargs):
        total_loss, should_stop = slim.learning.train_step(session, *args, **kwargs)
        if ( ts_fn.step % (ts_fn.decay_steps) == 0):
            accuracy = 0
            itr = val_dataset.num_samples//val_batch_size
            for i in range(itr):
                accuracy += session.run(ts_fn.val_accuracy)
            print ('Step %s - Loss: %.2f Accuracy: %.2f%%, Highest Accuracy : %.2f%%'
                   % (str(ts_fn.step).rjust(6, '0'), total_loss, accuracy *100/itr, ts_fn.highest*100/itr))
            acc = tf.Summary(value=[tf.Summary.Value(tag="Accuracy", simple_value=accuracy*100/itr)])
            ts_fn.eval_writer.add_summary(acc, ts_fn.step)
            
            if accuracy > ts_fn.highest:
                ts_fn.saver.save(session, "%s/best_model.ckpt"%train_dir)
                print ('save new parameters')
                ts_fn.highest = accuracy
            
        ts_fn.step += 1
        return [total_loss, should_stop] 
    
    ts_fn.saver = tf.train.Saver()
    ts_fn.eval_writer = tf.summary.FileWriter('%s/eval'%train_dir,graph,flush_secs=60)
    ts_fn.step = 0
    ts_fn.decay_steps = decay_steps
    ts_fn.val_accuracy = val_accuracy
    ts_fn.highest = 0
    ts_fn.loss2 = loss2
    config = ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
#    config.gpu_options.visible_device_list = str(gpu_num)
    
    slim.learning.train(train_op, logdir = train_dir, global_step = global_step,
                        session_config = config,
                        init_fn=_get_init_fn(checkpoint_path, cpt_scopes, ignore_missing_vars),
                        summary_op = summary_op,
                        train_step_fn=ts_fn,
                        number_of_steps = max_number_of_steps,
                        log_every_n_steps =  40,                #'The frequency with which logs are print.'
                        save_summaries_secs = 60,                #'The frequency with which summaries are saved, in seconds.'
                        save_interval_secs = 0)               #'The frequency with which the model is saved, in seconds.'


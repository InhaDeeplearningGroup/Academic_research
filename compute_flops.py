def conv_flops_params(total_flops, total_params, input_shape, conv_filter, 
                      padding = None, stride = 1, activation = 'relu'):
    if padding == None:
        padding = conv_filter[0]//2
    
    n = conv_filter[0] * conv_filter[1] * conv_filter[2]  # vector_length
    flops_per_instance = n + (n-1)    # general defination for number of flops (n: multiplications and n-1: additions)
    
    num_instances_per_filter = (( input_shape[0] - conv_filter[0] + 2*padding) / stride ) + 1  # for rows
    num_instances_per_filter *= (( input_shape[0] - conv_filter[0] + 2*padding) / stride ) + 1 # multiplying with cols
    
    flops_per_filter = num_instances_per_filter * flops_per_instance
    total_flops_per_layer = flops_per_filter * conv_filter[3]    # multiply with number of filters
    
    if activation == 'relu':
        # Here one can add number of flops required
        # Relu takes 1 comparison and 1 multiplication
        # Assuming for Relu: number of flops equal to length of input vector
        total_flops_per_layer += conv_filter[3]*input_shape[0]*input_shape[1]
    output_shape = (input_shape[0]//stride,input_shape[1]//stride,conv_filter[3])
    total_params_per_layer = n*conv_filter[3]+conv_filter[3]
    print('input shape : %s, output_shape : %s'%(input_shape, output_shape))
    print('flops : %d, params : %d'%(total_flops_per_layer,total_params_per_layer))
    total_flops  += total_flops_per_layer
    total_params += total_params_per_layer
    return total_flops, total_params, output_shape

def pool_flops_params(total_flops, total_params, input_shape, stride = 2):
    total_flops += input_shape[0]*input_shape[1]*input_shape[2]*3/4
    input_shape = (input_shape[0],input_shape[1]//stride,input_shape[2]//stride)
    
    return total_flops, total_params, input_shape

tf = 0
tp = 0

input_shape = (32,32,3)
tf, tp, input_shape = conv_flops_params(tf, tp, input_shape, (3,3,3,64), stride = 2)
tf, tp, input_shape = conv_flops_params(tf, tp, input_shape, (3,3,64,128), stride = 2)
tf, tp, input_shape = conv_flops_params(tf, tp, input_shape, (3,3,128,256), stride = 2)
tf, tp, input_shape = conv_flops_params(tf, tp, input_shape, (3,3,256,512), stride = 2)
input_shape = (1,1,input_shape[0]*input_shape[1]*input_shape[2])
tf, tp, input_shape = conv_flops_params(tf, tp, input_shape, (1,1,2048,1024), padding = 0,stride = 1)
tf, tp, input_shape = conv_flops_params(tf, tp, input_shape, (1,1,1024,100), padding = 0,stride = 1)

print ('total_flops : %d, total_params : %d'%(tf,tp))

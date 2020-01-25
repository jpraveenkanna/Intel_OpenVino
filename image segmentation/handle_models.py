import cv2
import numpy as np


def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # TODO 1: Extract only the second blob output (keypoint heatmaps)
    #print(output.keys())
    #o/p dict_keys(['Mconv7_stage2_L1', 'Mconv7_stage2_L2'])
    #print(output['Mconv7_stage2_L2'].shape)
    #o/p (1, 19, 32, 57)
    heatmaps = output['Mconv7_stage2_L2']
    # TODO 2: Resize the heatmap back to the size of the input
    #print(heatmaps.shape[1]) ===> 19
    #print(input_shape[0]) ===> 750
    #print(input_shape[1]) ===> 1000
    #print(len(heatmaps[0])) ===> 19, heatmaps[1] ===> index 1 is out of bounds for axis 0 with size 1
    out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    for h in range(len(heatmaps[0])):
        out_heatmap[h] = cv2.resize(heatmaps[0][h], input_shape[0:2][::-1])

    return out_heatmap


def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    '''
    # TODO 1: Extract only the first blob output (text/no text classification)
    #print(output.keys()) ===> dict_keys(['model/segm_logits/add', 'model/link_logits_/add'])
    text_class = output['model/segm_logits/add']
    #print(output['model/segm_logits/add'].shape) ===>(1, 2, 192, 320)
    # TODO 2: Resize this output back to the size of the input
    out_text = np.zeros([text_class.shape[1], input_shape[0], input_shape[1]])
    #print(text_class.shape[1]) ===> 2
    #print(len(text_class[0])) ===> 2
    #print(text_class[0][0].shape) ===> (192, 320)
    #print(input_shape[0:2][::-1]) ===> (1000, 667)
    #print(input_shape[0:2])  ===> (667, 1000)
    #cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    for h in range(len(text_class[0])):
        out_text[h] = cv2.resize(text_class[0][h], input_shape[0:2][::-1])
    return out_text


def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    # TODO 1: Get the argmax of the "color" output
    color = output['color'].flatten()
    car_type = output['type'].flatten()
    #print(color.shape)
    #print(color)
    #print(car_type.shape)
    #print(car_type)
    color_class = np.argmax(color)
    # TODO 2: Get the argmax of the "type" output
    type_class = np.argmax(car_type)
    return color_class, type_class

def handle_SEG(output, input_shape):
    segm_img = output['argmax']
    classes = cv2.resize(segm_img[0].transpose((1,2,0)),(input_shape[1],input_shape[0]),interpolation=cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)
    out_mask = classes * (255/20)
    return out_mask
    
    '''
    #print(output.keys()) ----> dict_keys(['argmax'])
    #print(type(output)) ----> <class 'dict'>
    
    #print(segm_img.shape) ---> (1, 1, 1024, 2048) ==> outputs a blob with the shape [B, H=1024, W=2048]
    #print(len(input_shape)) ---> 3
    #print(input_shape) #----> (562, 1000, 3)
    
    #print(segm_img[0])
    #print(len(segm_img)) ---> 1
    #print(type(segm_img)) ---><class 'numpy.ndarray'>
    
    #print(segm_img[0].shape) ---> (1, 1024, 2048)
    
    #print(classes.shape) #---> (562, 1000)
    
    print(unique_classes) #--> [ 0.  2.  3.  4.  8.  9. 13.] --> 13 represent car
    print(np.argmax(unique_classes))
    
    unique, counts = np.unique(classes, return_counts=True)
    #print(type(counts)) --> <class 'numpy.ndarray'>
    #print(np.asarray((unique, counts)).T)
    filter_img = np.where(classes != 13, 0, 13)
    print(np.unique(filter_img)) #--> [0 8]
    filter_mask = filter_img * (255/20)
    #print(filter_mask.shape)
    

    return out_mask
'''

def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    elif model_type == "IMG_SEG":
        return handle_SEG
    else:
        return None


'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
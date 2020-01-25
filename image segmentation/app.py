import argparse
import cv2
import numpy as np

from handle_models import handle_output, preprocessing
from inference import Network


CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
CAR_TYPES = ["car", "bus", "truck", "van"]


def get_args():
    '''
    Gets the arguments from the command line.
    '''

    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
    # -- Create the descriptions for the commands

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML file"
    t_desc = "The type of model: POSE, TEXT or CAR_META"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-t", help=t_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()

    return args


def get_mask(processed_output):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((processed_output, empty, processed_output))

    return mask
def mask(processed_output):
    empty = np.zeros(processed_output.shape)
    processed_output1 = processed_output * (255/100)
    processed_output2 = processed_output * (255/50)
    processed_output3 = processed_output * (255/150)  
    
    mask = np.dstack((processed_output1,processed_output1,processed_output3))
    #print(mask.shape) -->(750,1000,3)
    #print(processed_output.shape) --> (750,1000)
    #converted_img = cv2.cvtColor(processed_output, cv2.COLOR_GRAY2BGR)
    #print(converted_img.shape) --> (750,1000,3)
    mask_gray = cv2.normalize(src=mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    mask_colored_normalization = cv2.applyColorMap(mask_gray, cv2.COLORMAP_HSV)

    img = np.uint8(processed_output)
    mask_colored = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
    
    return mask_colored

def create_output_image(model_type, image, output):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''
    if model_type == "POSE":
        # Remove final part of output not used for heatmaps
        output = output[:-1]  #The first blob contains keypoint pairwise relations (part affinity fields), the second one contains keypoint heatmaps.
        # Get only pose detections above 0.5 confidence, set to 255
        for c in range(len(output)):
            output[c] = np.where(output[c]>0.5, 255, 0) #while true returns 255 else 0
        # Sum along the "class" axis
        output = np.sum(output, axis=0)
        # Get semantic mask
        pose_mask = get_mask(output)
        # Combine with original image
        image = image + pose_mask
        return image
    elif model_type == "TEXT":
        # Get only text detections above 0.5 confidence, set to 255
        output = np.where(output[1]>0.5, 255, 0)
        # Get semantic mask
        text_mask = get_mask(output)
        # Add the mask to the image
        image = image + text_mask
        return image
    elif model_type == "CAR_META":
        # Get the color and car type from their lists
        color = CAR_COLORS[output[0]]
        car_type = CAR_TYPES[output[1]]
        # Scale the output text by the image shape
        scaler = max(int(image.shape[0] / 1000), 1)
        # Write the text of color and type onto the image
        image = cv2.putText(image, 
            "Color: {}, Type: {}".format(color, car_type), 
            (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 
            2 * scaler, (255, 255, 255), 3 * scaler)
        return image
    elif model_type == "IMG_SEG":
        #output = np.where(output[1]>0.5, 255, 0)
        img_mask = mask(output)
        image = image + img_mask
        return img_mask
    else:
        print("Unknown model type, unable to create output image.")
        return image


def perform_inference(args):
    '''
    Performs inference on an input image, given a model.
    '''
    # Create a Network for using the Inference Engine
    inference_network = Network()
    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)

    # Read the input image
    image = cv2.imread(args.i)

    ### TODO: Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)
    #print(preprocessed_image.shape())

    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = inference_network.extract_output()

    ### TODO: Handle the output of the network, based on args.t
    ### Note: This will require using `handle_output` to get the correct
    ###       function, and then feeding the output to that function.
    
    process_func = handle_output(args.t)
    processed_output = process_func(output, image.shape)

    # Create an output image based on network
    try:
        output_image = create_output_image(args.t, image, processed_output)
        #print("Processed output:",processed_output)
        print("Success")
    except Exception as e:
        output_image = image
        print("Error \n"+ str(e))

    # Save down the resulting image
    cv2.imwrite("outputs/{}-output.png".format(args.t), output_image)


def main():
    args = get_args()
    perform_inference(args)


if __name__ == "__main__":
    main()


## TODOs

In `handle_models.py`, you will need to implement `handle_pose`, `handle_text`, and
`handle_car`.

In `app.py`, first, you'll need to use the input shape of the network to call the `preprocessing`
function. Then, you need to call `handle_output` with the appropriate model argument 
in order to get the right handling function. With that function, you can then feed the output
of the inference request in in order to extract the output. 

Note that there is some additional post-processing done for you in `create_output_image`
within `app.py` to help display the output back onto the input image.

## Testing the apps

To test your implementations, you can use `app.py` to run each edge application, with
the following arguments:
- `-t`: The model type,  which should be one of `"POSE"`, `"TEXT"`, or `"CAR_META"`
- `-m`: The location of the model .xml file
- `-i`: The location of the input image used for testing
- `-c`: A CPU extension file, if applicable. See below for what this is for the workspace.
The results of your output will be saved down for viewing in the `outputs` directory.

As an example, here is an example of running the app with related arguments:

```
python app.py -i "images/blue-car.jpg" -t "CAR_META" -m "/home/workspace/models/vehicle-attributes-recognition-barrier-0039.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
```

## Model Documentation

Once again, here are the links to the models, so you can use the **Output** section to help
you get started (there are additional comments in the code to assist):

- Human Pose Estimation: [human-pose-estimation-0001](https://docs.openvinotoolkit.org/latest/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html)
- Text Detection: [text-detection-0004](http://docs.openvinotoolkit.org/latest/_models_intel_text_detection_0004_description_text_detection_0004.html)
- Determining Car Type & Color: [vehicle-attributes-recognition-barrier-0039](https://docs.openvinotoolkit.org/latest/_models_intel_vehicle_attributes_recognition_barrier_0039_description_vehicle_attributes_recognition_barrier_0039.html)

# Project Write-Up

The people counter application is a smart video IoT solution that can detect people in a designated area of observation,
 providing the number of people in the frame, average duration of people in frame, 
 total count of people since the start of the observation session and an alarm that sends an alert to the 
 UI telling the user when a person enters the video frame.

In this project I used 'person-detection-retail-0013'. This model was taken from website:
https://docs.openvinotoolkit.org/2019_R3/_models_intel_index.html

This model give me the best results compared to other models:

    -ssd_mobilenet_v2_coco_2018_03_29
    -faster_rcnn_inception_v2_coco_2018_01_28

FROM:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

## Explaining Custom Layers

- Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.
- Model Optimizer searches for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation.
- The list of known layers is different for each of supported frameworks. To see the layers supported by your framework, refer to [Supported Framework Layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html).

    #### In this project:
        - I didn't use any supported layers
    
## Assess Model Use Cases

Some of the potential use cases of the people counter app are

  - Visitor analysis for shops and moles
  - it can be modified to detect people trying to inter restricted area
  - It can be modified to notify people about social distance to prevent spread of coronavirus or any case similar 
  - It can be modified to notify people about masks to prevent spread of coronavirus or any case similar

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

Better the model accuracy, More are the chances to obtain the desired results through an app deployed at edge.

Focal length/image also have a effect as better be the pixel quality of image or better the camera focal length,more clear results ww will obtain.

Lighter the model, More faster it will get execute and more adequate results in faster time as compared to a heavier model.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [ssd_mobilenet_v2_coco]
  - [detection_model_zoo]
  - Location on workspace
    - /home/workspace/ssd_mob
    
  - I converted the model to an Intermediate Representation with the following arguments...
    - cd /opt/intel/openvino/deployment_tools/model_optimizer
    - python mo_tf.py --input_model /home/workspace/ssd_mob/frozen_inference_graph.pb --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/workspace/ssd_mob/pipeline.config --reverse_input_channels -o /home/workspace/ssd_mod_2
    
  - I used this model using
    - python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mod/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    
  - Execution Attributes
    - FPS: 37.67
    - Latency (ms): 26.61
    - Total Execution Time (ms): 20043.8

  - The model was insufficient for the app because...
    - Total Count -> 35
    - Average Duration -> 00:02
    - Inference Time -> 70
  
- Model 2: [ssdlite_mobilenet_v2_coco]
  - [detection_model_zoo]
  - Location on workspace
    - /home/workspace/ssd_mob_2
    
  - I converted the model to an Intermediate Representation with the following arguments...
    - cd /opt/intel/openvino/deployment_tools/model_optimizer
    - python mo_tf.py --input_model /home/workspace/ssd_mob_2/frozen_inference_graph.pb --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/workspace/ssd_mob_2/pipeline.config --reverse_input_channels -o /home/workspace/ssd_mod_2  
    
  - I used this model using
    - python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mod_2/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

  - Execution Attributes
    - FPS: 80.1
    - Latency (ms): 12.52
    - Total Execution Time (ms): 20013.21
    
  - The model was insufficient for the app because...
    - Total Count -> 37
    - Average Duration -> 00:01
    - Inference Time -> 30

- Model 3: [ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync]
  - [detection_model_zoo]
  - Location on workspace
    - /home/workspace/ssd_mob_depth
    
  - I converted the model to an Intermediate Representation with the following arguments...
    - cd /opt/intel/openvino/deployment_tools/model_optimizer
    - python mo_tf.py --input_model /home/workspace/ssd_mob_depth/frozen_inference_graph.pb --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/workspace/ssd_mob_depth/pipeline.config --reverse_input_channels -o /home/workspace/ssd_mod_depth
    
  - I used this model using
    - python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mod_depth/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

  - Execution Attributes
    - FPS: 58.6
    - Latency (ms): 17.51
    - Total Execution Time (ms): 20016.58
    
  - The model was insufficient for the app because...
    - Total Count -> 64
    - Average Duration -> 00:00
    - Inference Time -> 26
    
- Model 4: [faster_rcnn_inception_v2_coco]
  - [detection_model_zoo]
  - Location on workspace
    - /home/workspace/rcnn_mob
    
  - I converted the model to an Intermediate Representation with the following arguments...
    - cd /opt/intel/openvino/deployment_tools/model_optimizer
    - python mo_tf.py --input_model /home/workspace/rcnn_mob/frozen_inference_graph.pb --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/workspace/rcnn_mob/pipeline.config --reverse_input_channels -o /home/workspace/rcnn_mod

  - I used this model using
    - python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m rcnn_mod/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

  - Execution Attributes
    - FPS: 2.39
    - Latency (ms): 424.53
    - Total Execution Time (ms): 20532.51
    
  - The model was insufficient for the app because...
    - Total Count -> 35
    - Average Duration -> 00:02
    - Inference Time -> 70

  - # Trying to fix the model....
    - I used this command.......
    -  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m rcnn_mod/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    - The model output is:
      - Total Count -> 6
      - Average Duration -> 00:17
      - Inference Time -> 0
      - ### Numbers looks good but the app run very slow

- Model 5: [person-detection-retail-0013]
  - [OpenVINO™ Toolkit Pre-Trained Models]
  - Location on workspace
    - /home/workspace/person_mod

  - I downloaded the model with the following arguments...
    - cd /opt/intel/openvino/deployment_tools/tools/model_downloader
    - sudo ./downloader.py --name person-detection-retail-0013 --precisions FP16 -o /home/workspace/person_mod
  
  - I used this command to run the app
    - cd /home/workspace
    - python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person_mod/person-detection-retail-0013.xml  -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

  - Execution Attributes
    - FPS: 53.57
    - Latency (ms): 18.43
    - Total Execution Time (ms): 20012.96
  
  - ##### The model was sufficient for the app
    - Total Count -> 11
    - Average Duration -> 00:10
    - Inference Time -> 47
    
  - # Trying to fix the model....
    - I used this command.......
    -  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person_mod/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    - The model output is:
      - Total Count -> 9
      - Average Duration -> 00:13
      - Inference Time -> 0
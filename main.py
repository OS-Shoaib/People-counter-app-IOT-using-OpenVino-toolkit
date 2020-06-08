import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    @return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    """
    Connect to the MQTT client
    @return: client
    """

    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def draw_boxes(frame, result, width, height, prob_threshold):
    """
    Draw bounding boxes onto the frame.
    @param frame: frame from camera/video
    @param result: list contains the data to parse ssd
    @param width: box width
    @param height: box height
    @param prob_threshold:Probability threshold
    @return: person count and frame
    """
    counter = 0
    for box in result[0][0]:
        conf = box[2]
        if conf >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            counter += 1
    return frame, counter


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    @param args: Command line arguments parsed by `build_argparser()`
    @param client: MQTT client
    @return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Initial Parameters
    single_image_mode = False
    start_time = time.time()
    request_id = 0
    total_count = 0
    last_count = 0

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args.device, request_id, args.cpu_extension)
    input_shape = infer_network.get_input_shape()

    # Handle the input stream
    if args.input.endswith('.jpg') or args.input.endswith('.bmp'):  # Image mode
        single_image_mode = True
        input_stream = args.input

    elif args.input == 'CAM':  # Camera mode
        input_stream = 0

    else:  # Input is a video path
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Loop until stream is over
    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()

        if not flag:
            break

        key_pressed = cv2.waitKey(60)
        # Pre-process the image as needed
        image = cv2.resize(frame, (input_shape[3], input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        # Detect Time
        infer_start = time.time()
        # Start asynchronous inference for specified request
        infer_network.exec_net(request_id, image)

        # Wait for the result
        if infer_network.wait(request_id) == 0:
            # Detect Time
            detect_time = time.time() - infer_start
            # Get the results of the inference request
            result = infer_network.get_output(request_id)
            # Get Inference Time
            infer_time = "Inference Time: {:.3f}ms".format(detect_time * 1000)
            # Extract any desired stats from the results
            frame, counter = draw_boxes(frame, result, width, height, prob_threshold)

            # Get a writen text on the video
            cv2.putText(frame, "Counted Number: {} ".format(counter),
                        (20, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, infer_time, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

            # Calculate and send relevant information on
            if counter > last_count:
                start_time = time.time()
                # current_count, total_count and duration to the MQTT server
                total_count = total_count + counter - last_count
                # Topic "person": keys of "count" and "total"
                client.publish("person", json.dumps({"total": total_count}))

            # Topic "person/duration": key of "duration"
            if counter < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                # Publish messages to the MQTT server
                client.publish("person/duration", json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": counter, "total": total_count}))
            last_count = counter

            # Break if escape key pressed
            if key_pressed == 27:
                break

        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        # Write an output image if `single_image_mode`
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.
    @return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()

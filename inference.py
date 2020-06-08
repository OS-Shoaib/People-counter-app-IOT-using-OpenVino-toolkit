#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        """
        Initialize any class variables desired
        """
        self.network = None
        self.plugin = None
        self.input_blob = None
        self.output_blob = None
        self.network_plugin = None
        self.inference_request = None

    def load_model(self, model, device, num_requests, cpu_extension=None, plugin=None):
        """
        Load the model
        @param model: .xml file of a pre trained model
        @param device: The target device
        @param num_requests: Number of frequency requests to the Inference Engine
        @param cpu_extension: Extension for CPU device
        @param plugin: Plugin for specific device
        @return: The loaded inference plugin
        """
        # Load the model
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Plugin initialization for specified device
        if not plugin:
            log.info("Initializing plugin for {} device...".format(device))
            self.plugin = IECore()
        else:
            self.plugin = plugin

        # load cpu extension if specified
        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)
        
        # Read IR
        log.info("Reading IR...")
        self.network = IENetwork(model=model_xml, weights=model_bin)
        log.info("Loading IR to the plugin...")
       
        if self.plugin.device == "CPU":
            # Get the supported layers of the network
            supported_layers = self.plugin.get_supported_layers(self.network)

            # Check for any unsupported layers, and let the user
            # know if anything is missing. Exit the program, if so.
            unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
            if len(unsupported_layers) != 0:
                print("Unsupported layers found: {}".format(unsupported_layers))
                print("Check whether extensions are available to add to IECore.")
                exit(1)

        if num_requests == 0:
            # Load the network into the Inference Engine
            self.network_plugin = self.plugin.load(self.network)
        else:
            self.network_plugin = self.plugin.load(self.network, num_requests=num_requests)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        """
        @return: The shape of the input layer
        """
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, request_id, frame):
        """
        Start an asynchronous request
        @return: Necessary information
        """

        self.inference_request = self.network_plugin.start_async(
            request_id=request_id, inputs={self.input_blob: frame}
        )

        return self.network_plugin

    def wait(self, request_id):
        """
        Wait for the request to be complete
        @return: Necessary information
        """
        request_waiting = self.network_plugin.requests[request_id].wait(-1)
        return request_waiting

    def get_output(self, request_id, output=None):
        """
        Extract and return the output results
        @return: The output results
        """
        if output:
            output = self.inference_request.outputs[output]
        else:
            output = self.network_plugin.requests[request_id].outputs[self.output_blob]
        return output

    def clean(self):
        """
        Deletes all the instances
        @return: None
        """
        del self.network_plugin
        del self.plugin
        del self.network

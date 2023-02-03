//
//  Copyright 2023 Brian Keith Smith
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  Created by Brian Smith on 1/26/23.
//

import XCTest
import MLCompute
import MimicTransferables
@testable import MimicComputeEngine

final class MlcLayerConversionsTests: XCTestCase {
    let label = "TestLayer"
    
    func testFullyConnectedLayerConversion() throws {
        let inputChannels = 4
        let outputChannels = 2
        
        let weights = Tensor([[[Float]]]([[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]]))
        let biases = Tensor([[Float]]([[0.4, 0.2]]))
        let descriptor = MLCConvolutionDescriptor(kernelSizes: (inputChannels, outputChannels),
                                                  inputFeatureChannelCount: inputChannels,
                                                  outputFeatureChannelCount: outputChannels)
        let mlcLayer = try XCTUnwrap(MLCFullyConnectedLayer(weights: weights.makeMlcTensor(),
                                                            biases: biases.makeMlcTensor(),
                                                            descriptor: descriptor))
        mlcLayer.label = label
        let layer = try mlcLayer.makeLayer()
        
        XCTAssertEqual(layer.label, label)
        XCTAssertEqual(layer.dataType, .float32)
        XCTAssertEqual(layer.kind, .fullyConnected)
        XCTAssertEqual(layer.inputFeatureChannelCount, inputChannels)
        XCTAssertEqual(layer.outputFeatureChannelCount, outputChannels)
        XCTAssertEqual(layer.weights, Tensor(weights, shape: Array(weights.shape[1...])))
        XCTAssertEqual(layer.biases, Tensor(biases, shape: Array(biases.shape[1...])))
    }
    
    func testConvolutionLayerConversion() throws {
        let featureChannels = 2
        let kernelHeight = 2
        let kernelWidth = 2
        let outputChannels = 2
        
        let weights = Tensor([[[[Float]]]]([
            [
                // output channel 0
                // feature channel 0
                [
                    [0.1, 0.2],
                    [0.1, 0.2]
                ],
                // feature channel 1
                [
                    [0.1, 0.2],
                    [0.1, 0.2]
                ],
                // output channel 1
                // feature channel 0
                [
                    [0.1, 0.2],
                    [0.1, 0.2]
                ],
                // feature channel 1
                [
                    [0.1, 0.2],
                    [0.1, 0.2]
                ]
            ]
        ]),
                             featureChannelPosition: .first)
        let biases = Tensor([[Float]]([[0.3, 0.1]]))
        let descriptor = MLCConvolutionDescriptor(transposeWithKernelWidth: kernelWidth,
                                                  kernelHeight: kernelHeight,
                                                  inputFeatureChannelCount: featureChannels,
                                                  outputFeatureChannelCount: outputChannels)
        let mlcLayer = try XCTUnwrap(MLCConvolutionLayer(weights: weights.makeMlcTensor(),
                                                      biases: biases.makeMlcTensor(),
                                                      descriptor: descriptor))
        mlcLayer.label = label
        let layer = try mlcLayer.makeLayer()
        
        XCTAssertEqual(layer.label, label)
        XCTAssertEqual(layer.dataType, .float32)
        XCTAssertEqual(layer.kind, .convolution)
        XCTAssertEqual(layer.kernelSize, Layer.KernelSize(height: kernelHeight, width: kernelWidth))
        XCTAssertEqual(layer.inputFeatureChannelCount, featureChannels)
        XCTAssertEqual(layer.outputFeatureChannelCount, outputChannels)
        XCTAssertEqual(layer.weights, Tensor(weights,
                                             shape: [outputChannels, featureChannels, kernelHeight, kernelWidth]))
        XCTAssertEqual(layer.biases, Tensor(biases, shape: Array(biases.shape[1...])))
    }
}

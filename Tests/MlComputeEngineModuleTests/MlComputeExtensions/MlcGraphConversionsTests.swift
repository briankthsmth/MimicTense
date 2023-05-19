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
//  Created by Brian Smith on 2/1/23.
//

import XCTest
import MLCompute
import MimicTransferables
@testable import MlComputeEngineModule

final class MlcGraphConversionsTests: XCTestCase {
    func testGraphConversion() throws {
        let layerDescriptor = MLCConvolutionDescriptor(kernelSizes: (3, 2),
                                                       inputFeatureChannelCount: 3,
                                                       outputFeatureChannelCount: 2)
        
        let source = Tensor(shape: [1, 3], dataType: .float32)
        let weights = Tensor([[[Float]]]([[[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]]]))
        let biases = Tensor([[Float]]([[0.6, 0.3]]))
        let platformGraph = MLCGraph()
        let platformLayer = try XCTUnwrap( MLCFullyConnectedLayer(weights: weights.makeMlcTensor(),
                                                                  biases: biases.makeMlcTensor(),
                                                                  descriptor: layerDescriptor))
        let _ = platformGraph.node(with:platformLayer, source: source.makeMlcTensor())
        let graph = try platformGraph.makeGraph()
        
        XCTAssertEqual(graph.kind, .sequential)
        XCTAssertEqual(graph.dataType, .float32)
        XCTAssertEqual(graph.inputTensors, [[source]])
        XCTAssertEqual(graph.layers.count, 1)
        XCTAssertEqual(graph.layers.first?.kind, .fullyConnected)
        XCTAssertEqual(graph.featureChannelPosition, .first)
    }
    
    func testGraphConversionFeatureChannelLast() throws {
        // MLCompute only supports the input feature channels first, so to support
        // it last transpose layers are added to reshape the inputs and outputs to
        // the first position and back.
        let inputFeatureChannels = 2
        let outputFeatureChannels = 2
        let kernelHeight = 1
        let kernelWidth = 1
        let layerDescriptor = MLCConvolutionDescriptor(transposeWithKernelWidth: kernelWidth,
                                                       kernelHeight: kernelHeight,
                                                       inputFeatureChannelCount: inputFeatureChannels,
                                                       outputFeatureChannelCount: inputFeatureChannels)
        let input = Tensor(shape: [1, 4, 4, 2], dataType: .float32, featureChannelPosition: .last)
        let weights = MLCTensor(shape: [
            1, // batch size = 1
            outputFeatureChannels * inputFeatureChannels,
            kernelHeight,
            kernelWidth ],
                                randomInitializerType: .uniform)
        let biases = MLCTensor(shape: [1, 2], randomInitializerType: .uniform)
        
        let toFirstTransposeLayer = try XCTUnwrap(MLCTransposeLayer(dimensions: [0, 3, 1, 2]))
        let convolutionLayer = try XCTUnwrap(MLCConvolutionLayer(weights: weights,
                                                                 biases: biases,
                                                                 descriptor: layerDescriptor))
        let toLastTransposeLayer = try XCTUnwrap(MLCTransposeLayer(dimensions: [0, 2, 3, 1]))
        let platformGraph = MLCGraph()
        let toFirstOutput = try XCTUnwrap(platformGraph.node(with: toFirstTransposeLayer, source: input.makeMlcTensor()))
        let convolutionOutput = try XCTUnwrap(platformGraph.node(with: convolutionLayer, source: toFirstOutput))
        let _ = try XCTUnwrap(platformGraph.node(with:toLastTransposeLayer, source:convolutionOutput))
        
        let graph = try platformGraph.makeGraph()

        XCTAssertEqual(graph.layers.count, 1)
        XCTAssertEqual(graph.layers.first?.kind, .convolution)
        XCTAssertEqual(graph.featureChannelPosition, .last)
    }
}

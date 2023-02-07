//
//  Copyright 2022 Brian Keith Smith
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
//  Created by Brian Smith on 5/20/22.
//

import XCTest
import MimicTransferables
@testable import MimicComputeEngine

class ConvolutionLayerInferenceGraphTests: XCTestCase {
    let dataTensorFeatureChannelLast = Tensor([[[[Float]]]]([[
        [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
        [[0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]
    ]]))
    let dataTensorFeatureChannelFirst = Tensor([[[[Float]]]]([[
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ],
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ],
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ],
    ]])
                                               , featureChannelPosition: .first)
    
    
    func testConvolutionLayerGraphWithFeatureChannelFirst() async throws {
        let expectedArray = [[[[Float]]]]([[
            [
                [0.3, 0.6, 0.9],
                [1.2, 1.5, 1.8]
            ]
        ]])
        try await runConvolutionLayerTest(input: dataTensorFeatureChannelFirst,
                                          expected: expectedArray)
    }
    
    func testConvolutionLayerGraphWithFeatureChannelLast() async throws {
        let expectedArray = [[[[Float]]]]([[
            [[0.3], [0.6], [0.9]],
            [[1.2], [1.5], [1.8]]
        ]])
        try await runConvolutionLayerTest(input: dataTensorFeatureChannelLast,
                                          expected: expectedArray)
    }
    
    private func runConvolutionLayerTest(input: Tensor, expected: [[[[Float]]]]) async throws {
        let batchSize = input.shape[0]
        let expectedOutputTensor = Tensor(expected)
        let dataSet = DataSet(inputTensor: input, batchSize: batchSize)
        let weightsTensor = Tensor(Float(1))
        let layer = Layer(kind: .convolution,
                          dataType: input.dataType,
                          kernelSize: Layer.KernelSize(height: 1, width: 1),
                          inputFeatureChannelCount: input.featureChannelCount,
                          outputFeatureChannelCount: 1,
                          weights: weightsTensor)
        let graph = Graph(kind: .sequential,
                          dataType: input.dataType,
                          inputTensor: Tensor(shape: input.shape,
                                              dataType: input.dataType,
                                              featureChannelPosition: input.featureChannelPosition),
                          layers: [layer],
                          featureChannelPosition: input.featureChannelPosition)
        let inferenceGraph = try MlComputeInferenceGraph(graphs: [graph])
        try inferenceGraph.compile(device: .gpu)
        let results = try await inferenceGraph.execute(inputs: dataSet.makeBatch(at: 0),
                                                        batchSize: batchSize)

        let expectedVector = Array(expected.joined().joined().joined())
        
        XCTAssertEqual(results[0].shape, expectedOutputTensor.shape)
        XCTAssertEqual(results[0].dataType, expectedOutputTensor.dataType)
        assertEqual(resultTensor: results[0], expectedVector: expectedVector)
    }
}

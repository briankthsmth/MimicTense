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
//  Created by Brian Smith on 7/20/22.
//

import XCTest
import MimicTransferables
@testable import MimicComputeEngine

final class FullyConnectedLayerInferenceGraphTests: XCTestCase {
    func testSimpleAdditionGraph() async throws {
        let inputTensor = Tensor([Float]([2, 2]))
        let dataSet = DataSet(inputTensor: inputTensor, batchSize: 1)
        let weights = Tensor([[Float]]([[1, 0.5]]))
        let biases = Tensor([Float]([2]))
        let layer = Layer(kind: .fullyConnected,
                          dataType: .float32,
                          inputFeatureChannelCount: 2,
                          outputFeatureChannelCount: 1,
                          weights: weights,
                          biases: biases)
        let graph = Graph(kind: .sequential,
                          dataType: .float32,
                          inputTensor: Tensor(shape: [1, 2], dataType: .float32),
                          layers: [layer],
                          featureChannelPosition: .notApplicable)
        let inferenceGraph = try MlComputeInferenceGraph(graphs: [graph])
        inferenceGraph.compile(device: .gpu)
        let results = await inferenceGraph.execute(inputs: dataSet.makeBatch(at: 0),
                                                        batchSize: 1)
        let resultTensor = results[0]
        let expectedVector: [Float] = [5]
        
        XCTAssertEqual(resultTensor.shape, [1, 1])
        XCTAssertEqual(resultTensor.dataType, .float32)
        assertEqual(resultTensor: resultTensor, expectedVector: expectedVector)
    }
}

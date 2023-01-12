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
//
//  Created by Brian Smith on 9/26/22.
//

import XCTest
@testable import MimicComputeEngine
import MimicTransferables

final class MlComputeTrainingGraphTests: XCTestCase {
    let expectedWeights: [Float] = [0.47]
    let batchSize: Int = 2
    var batchIterations: Int!
    
    let inputChannels = 1
    let outputChannels = 1
    
    let trainingSamples: [[Float]] = [
        [1.5],
        [2.3],
        [-4.4],
        [5.2],
        [-0.8],
        [2.7],
        [0.12],
        [-3.12],
        [2.8],
        [4.2]
    ]
    var trainingResults: [Float]!
    
    override func setUpWithError() throws {
        batchIterations = trainingSamples.count / batchSize
        
        trainingResults = trainingSamples.map {
            $0.enumerated().reduce(0) { $0 + $1.element * expectedWeights[$1.offset] }
        }
    }
    
    func testTraining() async throws {
        let weights = Tensor(shape: [outputChannels, inputChannels],
                             dataType: .float32,
                             randomInitializerType: .uniform)
        let layer = Layer(label: "TestLayer",
                          kind: .fullyConnected,
                          dataType: .float32,
                          inputFeatureChannelCount: inputChannels,
                          outputFeatureChannelCount: outputChannels,
                          weights: weights)
        let graph = Graph(kind: .sequential,
                          dataType: .float32,
                          inputTensor: Tensor(shape: [batchSize, inputChannels], dataType: .float32),
                          layers: [layer],
                          featureChannelPosition: .notApplicable)
        
        let trainingGraph = try MlComputeTrainingGraph(graphs: [graph],
                                                       lossLabelTensors: [Tensor(shape: [batchSize, outputChannels],
                                                                                 dataType: .float32)],
                                                       lossFunction: .meanSquaredError,
                                                       optimizer: .rootMeanSquare(learningRate: 0.01))
        try trainingGraph.compile(device: .cpu)
        for _ in 0 ..< 10 {
            for batch in 0 ..< batchIterations {
                let batchStart = 2 * batch
                let batchEnd = batchStart + batchSize
                let batchSamples = Array(trainingSamples[batchStart ..< batchEnd])
                let batchResults = Array(trainingResults[batchStart ..< batchEnd])
                try await trainingGraph.execute(inputs: [Tensor(batchSamples)],
                                                lossLables: [Tensor(batchResults)],
                                                batchSize: batchSize)
            }
        }
        let trainedWeights = try trainingGraph.copyWeights(for: layer)
        let trainedVector = try XCTUnwrap(trainedWeights.extract(Float.self) as? [[Float]])
        for (expectedWeight, weight) in zip(expectedWeights, trainedVector[0]) {
            XCTAssertEqual(weight, expectedWeight, accuracy: 0.01)
        }
    }
    
    func testCopyWeightsForLayer() throws {
        let weightsVector = [Float]([0.4, 0.8])
        let weights = Tensor(shape: [2, 1], data: weightsVector.makeBuffer(), dataType: .float32)
        let layer = Layer(label: "TestLayer",
                          kind: .fullyConnected,
                          dataType: .float32,
                          inputFeatureChannelCount: 2,
                          outputFeatureChannelCount: 1,
                          weights: weights)
        let graph = Graph(kind: .sequential,
                          dataType: .float32,
                          inputTensor: Tensor(shape: [1, 2], dataType: .float32),
                          layers: [layer],
                          featureChannelPosition: .notApplicable)
        let trainingGraph = try MlComputeTrainingGraph(graphs: [graph],
                                                       lossLabelTensors: [Tensor(shape: [1,1],
                                                                                 dataType: .float32)],
                                                       lossFunction: .meanSquaredError,
                                                       optimizer: .rootMeanSquare(learningRate: 0.01))
        try trainingGraph.compile(device: .gpu)
        
        let copiedWeights = try trainingGraph.copyWeights(for: layer)
        XCTAssertEqual(copiedWeights.shape, [1, 2])
        let copiedVector = try XCTUnwrap(copiedWeights.extract(Float.self) as? [[Float]])
        zip(copiedVector[0], weightsVector).forEach {
            XCTAssertEqual($0.0, $0.1)
        }
    }
}

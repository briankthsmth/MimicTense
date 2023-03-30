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
//  Created by Brian Smith on 2/10/23.
//

import XCTest
import MimicTense

final class TrainLinearRegressionTests: XCTestCase {
    struct Constant {
        static let slope: Float = 0.47
        static let intercept: Float = 0.7
        
        static let batchSize = 2
        static let learningRate: Float = 0.01
        static let epochs = 20
        
        static let layerName = "fullyConnected"
    }
    
    let inputs: [[Float]] = [
        [0.2],
        [1.3],
        [0.1],
        [-1.7],
        [0.6],
        [1.7],
        [0.4],
        [-0.45],
        [1.1],
        [-1.0]
    ]
    var labels: [[Float]]!
    var numberOfBatches: Int!
    
    override func setUpWithError() throws {
        labels = inputs.map { $0.map { Constant.slope * $0 + Constant.intercept } }
        numberOfBatches = inputs.count / Constant.batchSize
    }
    
    func testTrainLinearRegression() async throws {
        let train = try await Train<Float>(epochs: Constant.epochs,
                                           lossFunction: .meanSquaredError,
                                           optimizer: .rootMeanSquare(learningRate: Constant.learningRate))
        {
            TrainingDataSet(batchSize: Constant.batchSize) {
                TrainingData {
                    LabelData {
                        Tensor(labels)
                    }
                    InputData {
                        Tensor(inputs)
                    }
                }
            }
            Sequential<Float> {
                FullyConnected<Float>(name: Constant.layerName,
                                      weights: Tensor(shape: [1, 1], randomizer: .uniformDelayed),
                                      biases: Tensor([Float]([0])),
                                      inputFeatureChannelCount: 1,
                                      outputFeatureChannelCount: 1) {
                    Inputs {
                        Tensor<Float>(shape: [Constant.batchSize, 1])
                    }
                }
            }
        }
        .compile(device: .gpu)
        
        var outputCount = 0
        for try await batchOutput in train.outputStream {
            XCTAssertEqual(batchOutput.shape, [2, 1])
            outputCount += 1
        }
        XCTAssertEqual(outputCount, numberOfBatches * Constant.epochs)
        
        let weights = try train.retrieveWeights(for: Constant.layerName)
        let biases = try train.retrieveBiases(for: Constant.layerName)
        
        XCTAssertEqual(weights.rank2Data?.first?.first ?? 0, Constant.slope, accuracy: 0.01)
        XCTAssertEqual(biases.rank1Data?.first ?? 0, Constant.intercept, accuracy: 0.01)
    }
}

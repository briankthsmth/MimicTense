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
//  Created by Brian Smith on 3/16/23.
//

import XCTest
@testable import MimicTense

final class SessionRunnerTests: XCTestCase {
    struct Constant {
        static let slope: Float = 1.2
        static let intercept: Float = 0.5
        
        static let batchSize = 2
        static let epochs = 3
        
        static let layerName = "fully connected 1"
    }
    
    let inputs: [Float] = [
        1.2,
        3.4,
        -0.2,
        0.5,
        -1.9,
        2.3
    ]
    var labels: [Float]!
    var sessionRunner: SessionRunner<Float>!
    var batchCount: Int {
        inputs.count / Constant.batchSize
    }
    
    override func setUpWithError() throws {
        let labels = inputs.map { Constant.slope * $0 + Constant.intercept }
        let dataSet = TrainingDataSet(batchSize: Constant.batchSize) {
            TrainingData {
                LabelData {
                    Tensor(labels)
                }
                InputData {
                    Tensor(inputs)
                }
            }
        }
        let graph = Sequential<Float> {
            FullyConnected(name: Constant.layerName,
                           weights: Tensor(shape: [1, 1], randomizer: .uniformDelayed),
                           biases: Tensor([Float]([0])),
                           inputFeatureChannelCount: 1,
                           outputFeatureChannelCount: 1) {
                Inputs {
                    Tensor<Float>(shape: [Constant.batchSize, 1])
                }
            }
        }
        sessionRunner = try SessionRunner(kind: .training(lossFunction: .meanSquaredError,
                                                          optimizer: .rootMeanSquare(learningRate: 0.01)),
                                          epochs: Constant.epochs,
                                          dataSet: dataSet,
                                          graph: graph)
    }
    
    func testEpochs() async throws {
        try await sessionRunner.compile(device: .cpu)
        var outputCount: Int = 0
        for try await _ in sessionRunner.makeOutputStream() {
            outputCount += 1
        }
        XCTAssertEqual(outputCount / batchCount, Constant.epochs)
    }
    
    func testRetrieveLayer() throws {
        XCTAssertThrowsError(try sessionRunner.retrieveLayer(for: "No Name"))
        let layer = try sessionRunner.retrieveLayer(for: Constant.layerName)
        XCTAssertEqual(layer.label, Constant.layerName)
        XCTAssertEqual(layer.kind, .fullyConnected)
    }
}

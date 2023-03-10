//
//  TrainLinearRegressionTests.swift
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
    
    override func setUpWithError() throws {
        labels = inputs.map { $0.map { Constant.slope * $0 + Constant.intercept } }
    }
    
    func testTrainLinearRegression() async throws {
        let train = try await Train<Float>(lossFunction: .meanSquaredError,
                                    optimizer: .rootMeanSquare(learningRate: 0.01))
        {
            TrainingDataSet(batchSize: 2) {
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
                FullyConnected<Float>(weights: Tensor(shape: [1], randomizer: .uniformDelayed),
                                      biases: Tensor([Float]([0])),
                                      inputFeatureChannelCount: 1,
                                      outputFeatureChannelCount: 1) {
                    Inputs {
                        Tensor<Float>(shape: [2, 1])
                    }
                }
            }
        }
        .compile(device: .gpu)
        
    }
}

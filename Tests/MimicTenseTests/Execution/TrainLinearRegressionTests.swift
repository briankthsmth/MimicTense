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
        
        static let batchSize = 2
        static let learningRate: Float = 0.01
        static let epochs = 10
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
        let train = try await Train<Float>(lossFunction: .meanSquaredError,
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
        
        var outputCount = 0
        for try await outputStream in train.outputStream {
            XCTAssertEqual(outputStream.count, 1)
            let batchOutput = try XCTUnwrap(outputStream.first)
            XCTAssertEqual(batchOutput.shape, [2, 1])
            outputCount += 1
        }
        XCTAssertEqual(outputCount, numberOfBatches)
    }
}

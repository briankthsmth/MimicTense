//
//  TrainingDataTests.swift
//  
//
//  Created by Brian Smith on 3/2/23.
//

import XCTest
import MimicTense

final class TrainingDataTests: XCTestCase {
    func testResultBuilderInitialization() {
        let labelsTensor = Tensor([Float](arrayLiteral: 6, 7, 8))
        let trainingData = TrainingData {
            LabelData {
                labelsTensor
            }
            InputData {
                Tensor([Float](arrayLiteral: 1, 2, 3))
            }
            InputData {
                Tensor([Float](arrayLiteral: 3, 4, 5))
            }
        }
        XCTAssertEqual(trainingData.inputs.count, 2)
        XCTAssertEqual(trainingData.labels.data, labelsTensor)
    }
}

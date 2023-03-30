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

final class TrainingDataSetTests: XCTestCase {
    
    func testResultBuilderInitialization() throws {
        let trainingDataSet = TrainingDataSet<Float>(batchSize: 1) {
            TrainingData {
                LabelData {
                    Tensor([Float](arrayLiteral: 1, 2, 3))
                }
                InputData {
                    Tensor([Float](arrayLiteral: 1, 2, 3))
                }
            }
        }
        XCTAssertEqual(trainingDataSet.batchSize, 1)
        XCTAssertEqual(trainingDataSet.trainingData.inputs.count, 1)
        XCTAssertEqual(trainingDataSet.inputTensors.count, 1)
        let inputData = try XCTUnwrap(trainingDataSet.inputTensors.first)
        XCTAssertEqual(inputData.shape, [3])
        XCTAssertEqual(trainingDataSet.labels?.shape, [3])
    }
}

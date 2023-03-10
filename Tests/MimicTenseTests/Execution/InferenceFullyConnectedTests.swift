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
import MimicTense

final class InferenceFullyConnectedTests: XCTestCase {
    func testFullyConnectedGraph() async throws {
        let inference = try await Inference<Float> {
            InferenceDataSet(batchSize: 1) {
                InputData { Tensor<Float>([[2, 2]]) }
            }
            Sequential<Float> {
                FullyConnected(weights: Tensor<Float>([[1, 0.5]]),
                               biases: Tensor<Float>([1]),
                               inputFeatureChannelCount: 2,
                               outputFeatureChannelCount: 1) {
                    Inputs {
                        Tensor<Float>(shape: [1, 2])
                    }
                }
            }
        }
            .compile(device: .cpu)

        let expectedTensor = Tensor<Float>([[4]])
        var resultIndex = 0
        for try await outputTensors in inference.outputStream {
            XCTAssertEqual(outputTensors.count, 1)
            try XCTSkipUnless(outputTensors.count > 0, "Output tensors array was empty")
            
            let outputTensor = outputTensors[0]
            XCTAssertTrue(outputTensor.isEqual(expectedTensor, accuracy: 0.0001),
                          "\(outputTensor) is not equal to \(expectedTensor)")
            resultIndex += 1
        }
        XCTAssertEqual(resultIndex, 1)
    }
}

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
//  Created by Brian Smith on 4/27/22.
//

import XCTest
import MimicTense

class InferenceTests: XCTestCase {
    func testSimpleConvolutionGraph() async throws {
        let inference = try await Inference<Float> {
            ArrayDataSet(data: [[[[Float]]]]([[
                [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
                [[0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]
            ]]))
            Sequential<Float> {
                Convolution<Float>(kernelSize: KernelSize(height: 1, width: 1),
                                   inputFeatureChannelCount: 3,
                                   outputFeatureChannelCount: 1,
                                   weights: Tensor(1)) {
                    Inputs {
                        Tensor<Float>(shape: [1, 2, 3, 3])
                    }
                }
            }
        }
            .compile(device: .cpu)
    
        
        let expectedTensor = Tensor([[[[Float]]]]([[
            [[0.3], [0.6], [0.9]],
            [[1.2], [1.5], [1.8]]
        ]]))
        var resultIndex = 0
        for try await outputTensor in inference.outputStream {
            XCTAssertTrue(outputTensor.isEqual(expectedTensor, accuracy: 0.0001),
                          "\(outputTensor) is not equal to \(expectedTensor)")
            resultIndex += 1
        }
        XCTAssertEqual(resultIndex, 1)
    }
}

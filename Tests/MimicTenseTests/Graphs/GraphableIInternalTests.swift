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
//  Created by Brian Smith on 7/1/22.
//

import XCTest
import MimicTransferables
@testable import MimicTense

final class GraphableIInternalTests: XCTestCase {
    func testMakeTransferable() throws {
        let shape = [3, 4, 5, 5]
        let graph = try Sequential<Float> {
            Convolution<Float>(kernelSize: KernelSize(height: 2, width: 3),
                        inputFeatureChannelCount: 4,
                               outputFeatureChannelCount: 2) {
                Inputs {
                    MimicTense.Tensor<Float>(shape: shape)
                }
            }
        }
            .makeTransferable()
        
        XCTAssertEqual(graph.dataType, .float32)
        XCTAssertEqual(graph.kind, .sequential)
        XCTAssertEqual(graph.inputTensors, [[MimicTransferables.Tensor(shape: shape, dataType: .float32)]])
        XCTAssertEqual(graph.layers.count, 1)
        XCTAssertEqual(graph.featureChannelPosition, .last)
    }
}

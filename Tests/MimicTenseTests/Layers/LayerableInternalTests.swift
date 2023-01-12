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
//  Created by Brian Smith on 6/29/22.
//

import XCTest
import MimicTransferables
@testable import MimicTense

final class LayerableInternalTests: XCTestCase {
    func testMakeTransferable() throws {
        let layer = try? Convolution<Float>(kernelSize: KernelSize(height: 3, width: 3),
                                            inputFeatureChannelCount: 3,
                                            outputFeatureChannelCount: 1,
                                            weights: Tensor(Float(1)))
            .makeTransferable()
        XCTAssertEqual(layer?.dataType, .float32)
        XCTAssertEqual(layer?.kind, .convolution)
        XCTAssertNil(layer?.arithmeticOperation)
        XCTAssertEqual(layer?.kernelSize, MimicTransferables.Layer.KernelSize(height: 3, width: 3))
        XCTAssertEqual(layer?.inputFeatureChannelCount, 3)
        XCTAssertEqual(layer?.outputFeatureChannelCount, 1)
        XCTAssertEqual(layer?.weights, Tensor(Float(1)))
    }
}

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
//  Created by Brian Smith on 6/23/22.
//

import XCTest
import MimicTransferables
@testable import MimicTense

final class TensorInternalTests: XCTestCase {
    let vector = [Float]([1, 2, 3])

    func testMakeFromNeuralCoreTensor() throws {
        let transferableTensor = MimicTransferables.Tensor(vector)
        let tensor: MimicTense.Tensor<Float> = try MimicTense.Tensor<Float>.make(from: transferableTensor)
        XCTAssertEqual(tensor.shape, transferableTensor.shape)
        XCTAssertEqual(tensor.rank1Data, vector)
    }
    
    func testTranferableConversion() {
        let tensor = MimicTense.Tensor(vector)
        let expectectedTensor = MimicTransferables.Tensor(vector)
        let transferableTensor = try? tensor.makeTransferable()
        XCTAssertEqual(transferableTensor, expectectedTensor)
    }
    
    func testPlaceholderTransferableConversion() {
        let tensor = MimicTense.Tensor<Float>(shape: [3, 2])
        let expectedTensor = MimicTransferables.Tensor(shape: [3, 2], dataType: .float32)
        let transferableTensor = try? tensor.makeTransferable()
        XCTAssertEqual(transferableTensor, expectedTensor)
    }
}

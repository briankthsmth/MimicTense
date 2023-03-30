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

final class TensorInternalFactoryTests: XCTestCase {
    let vector = [Float]([1, 2, 3])

    func testMakeFromTransferableTensor() throws {
        let transferableTensor = MimicTransferables.Tensor(vector)
        let tensor: MimicTense.Tensor<Float> = try MimicTense.Tensor<Float>.make(from: transferableTensor)
        XCTAssertEqual(tensor.shape, transferableTensor.shape)
        XCTAssertEqual(tensor.rank1Data, vector)
    }
    
    func testTranferableConversion() throws {
        let tensor = MimicTense.Tensor(vector)
        let expectectedTensor = MimicTransferables.Tensor(vector)
        let transformedTensor = try tensor.makeTransferable()
        XCTAssertEqual(transformedTensor, expectectedTensor)
    }
    
    func testTransferableConversionWithPlaceholder() throws {
        let tensor = MimicTense.Tensor<Float>(shape: [3, 2])
        let expectedTensor = MimicTransferables.Tensor(shape: [3, 2], dataType: .float32)
        let transformedTensor = try? tensor.makeTransferable()
        XCTAssertEqual(transformedTensor, expectedTensor)
    }
    
    func testTransferableConversionWithRandomInitializer() throws {
        let tensor = MimicTense.Tensor<Float>(shape: [3, 2], randomizer: .uniformDelayed)
        let expectedTensor = MimicTransferables.Tensor(shape: [3, 2], dataType: .float32, randomInitializerType: .uniform)
        let transformedTensor = try tensor.makeTransferable()
        XCTAssertEqual(transformedTensor, expectedTensor)
    }
    
    func testMakeWithRandomData() throws {
        let shape = [2, 3]
        let tensor = MimicTense.Tensor<Float>.makeFillUniform(shape: shape, in: 0.0 ... 1.0)
        XCTAssertEqual(tensor.shape, shape)
        XCTAssertNotNil(tensor.rank2Data)
    }
}

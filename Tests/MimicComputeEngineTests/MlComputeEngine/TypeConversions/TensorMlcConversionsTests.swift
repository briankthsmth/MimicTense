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
//  Created by Brian Smith on 6/3/22.
//

import XCTest
import MimicTransferables
@testable import MimicComputeEngine
import MLCompute

class TensorMlComputeUtilitiesTests: XCTestCase {
    let vector: [Float] = [1, 2, 3, 4]
    var tensor: Tensor!
    
    override func setUpWithError() throws {
        let data = vector.makeBuffer()
        tensor = Tensor(shape: [4], data: data, dataType: .float32)
    }

    func testMakeMlcTensorWithoutData() {
        XCTAssertTrue(Tensor(shape: [2, 3], dataType: .float32).makeMlcTensor().comparePrototypeProperties(to: MLCTensor(shape: [2,3], dataType: .float32)))
        XCTAssertTrue(Tensor(shape: [1, 3, 2, 4], dataType: .float32, featureChannelPosition: .first).makeMlcTensor().comparePrototypeProperties(to: MLCTensor(shape: [1, 3, 2, 4], dataType: .float32)))
    }
    
    func testMakeMlcTensorWithData() {
        var vector: [Float] = vector
        let mlcTensorData = MLCTensorData(immutableBytesNoCopy: &vector,
                                          length: vector.count * MemoryLayout<Float>.size)
        let expectedTensor = MLCTensor(shape: [4], data: mlcTensorData, dataType: .float32)
        
        let mlcTensor = tensor.makeMlcTensor()
        XCTAssertTrue(mlcTensor.compare(to: expectedTensor))
    }
    
    func testMakeMlcTensorData() {
        let tensorData = tensor.makeMlcTensorData()
        XCTAssertEqual(tensorData.length, tensor.data.count)
        tensor.data.withUnsafeBytes {
            XCTAssertEqual(tensorData.bytes, $0.baseAddress)
        }
    }
}

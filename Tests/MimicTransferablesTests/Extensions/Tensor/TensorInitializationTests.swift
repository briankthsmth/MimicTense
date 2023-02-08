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
//  Created by Brian Smith on 1/25/23.
//

import XCTest
import MimicTransferables

final class TensorInitializationTests: XCTestCase {
    let data = TensorTestData()
    
    func testInitWithScalar() throws {
        let tensor = Tensor(data.scalar)
        XCTAssertEqual(tensor, data.scalarTensor)
    }
    
    func testInitWithVector() throws {
        let vector = [Float]([3, 4, 6, 7])
        let vectorData = vector.makeBuffer()
        let vectorTensor = Tensor(vector)
        let expectTensor = Tensor(shape: [vector.count], data: vectorData, dataType: .float32)
        XCTAssertEqual(vectorTensor, expectTensor)
    }
    
    func testInitWithRank4Tensor() throws {
        let tensor = Tensor([[[[Float]]]]([
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]
            ],
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]
            ]
        ]))
        let flattenedArray: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3 , 4, 5, 6, 7, 8, 9, 10, 11, 12]
        let expectedData = flattenedArray.makeBuffer()
        let expectedTensor = Tensor(shape: [2, 2, 2, 3], data: expectedData, dataType: .float32)
        XCTAssertEqual(tensor, expectedTensor)
    }
}

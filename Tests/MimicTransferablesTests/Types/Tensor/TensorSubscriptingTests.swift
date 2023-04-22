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

final class TensorSubscriptingTests: XCTestCase {
    let data = TensorTestData()
    
    func testSubscriptVectorInRange() {
        let tensor = Tensor([Float]([1, 2, 3, 4, 5]))
        let range = 1 ..< 4
        let closedRange = 1 ... 3
        let expectedTensor = Tensor([Float]([2, 3, 4]))
        XCTAssertEqual(tensor[range], expectedTensor)
        XCTAssertEqual(tensor[closedRange], expectedTensor)
    }
    
    func testSubscriptMatrixInRange() {
        let tensor = Tensor([[Float]]([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ]))
        let range = 2..<4
        let closedRange = 2...3
        let expectedTensor = Tensor([[Float]]([
            [7, 8, 9],
            [10, 11, 12]
        ]))
        XCTAssertEqual(tensor[range], expectedTensor)
        XCTAssertEqual(tensor[closedRange], expectedTensor)
    }
    
    func testSubscriptAtIndex() {
        let tensor = Tensor([Float]([1, 2, 3, 4, 5]))
        let scalarTensor = tensor[1]
        XCTAssertEqual(scalarTensor, Tensor([Float]([2])))
    }
}

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
//  Created by Brian Smith on 7/12/22.
//

import XCTest
import MimicTransferables

final class TensorArrayTests: XCTestCase {
    func testSubscriptVectors() {
        let vectorTensors = [Tensor]([
            Tensor([Float]([1, 2])),
            Tensor([Float]([3, 4])),
            Tensor([Float]([5, 6])),
            Tensor([Float]([7, 8]))
        ])
        let range = 1...2
        let expectedTensor = Tensor([[Float]]([
            [3, 4],
            [5, 6]
        ]))
        
        let tensorArray = TensorArray(tensors: vectorTensors)
        let tensor = tensorArray[range]
        XCTAssertEqual(tensor, expectedTensor)
    }
    
    func testSubscriptRank4Tensors() {
        let rank4Tensor = Tensor([[[[Float]]]]([
            [[[1], [1]]],
            [[[2], [2]]],
            [[[3], [3]]],
            [[[4], [4]]],
            [[[5], [5]]]
        ]))
        let range = 2...3
        let expectedTensor = Tensor([[[[Float]]]]([
            [[[3], [3]]],
            [[[4], [4]]]
        ]))
        let tensorArray = TensorArray(tensors: [rank4Tensor])
        let subtensor = tensorArray[range]
        XCTAssertEqual(subtensor, expectedTensor)
    }
    
    func testSubscriptRank4TensorsAcrossElements() {
        let tensors = [Tensor]([
            Tensor([[[[Float]]]]([
                [[[1]]],
                [[[2]]]
            ])),
            Tensor([[[[Float]]]]([
                [[[3]]],
                [[[4]]],
                [[[5]]]
            ]))
        ])
        let range = 1...2
        let tensorArray = TensorArray(tensors: tensors)
        let expectedTensor = Tensor([[[[Float]]]]([
            [[[2]]],
            [[[3]]]
        ]))
        let subTensor = tensorArray[range]
        XCTAssertEqual(subTensor, expectedTensor)
    }
}

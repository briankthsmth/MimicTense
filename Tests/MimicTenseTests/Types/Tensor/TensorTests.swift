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

class TensorTests: XCTestCase {
    let data4d = [[[[Float]]]]([[
        [[1, 2], [3,4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]]
    ]])
    var tensor4d: Tensor<Float>!
    
    override func setUpWithError() throws {
        tensor4d = Tensor(data4d)
    }
    
    func test4dTensor() {
        XCTAssertEqual(tensor4d.shape, [1, 3, 2, 2])
    }
    
    func test4dTensorSubscripting() {
        let subtensor3d = tensor4d[0]
        XCTAssertEqual(subtensor3d.shape, [3, 2, 2])
        XCTAssertEqual(subtensor3d.rank3Data, data4d[0])
        
        let tensorRow0 = subtensor3d[0]
        let tensorRow1 = subtensor3d[1]
        let tensorRow2 = subtensor3d[2]
        
        XCTAssertEqual(tensorRow0.shape, [2, 2])
        XCTAssertEqual(tensorRow0.rank2Data, data4d[0][0])
        XCTAssertEqual(tensorRow1.shape, [2, 2])
        XCTAssertEqual(tensorRow1.rank2Data, data4d[0][1])
        XCTAssertEqual(tensorRow2.shape, [2, 2])
        XCTAssertEqual(tensorRow2.rank2Data, data4d[0][2])
        
        let tensorRow0Column0 = tensorRow0[0]
        let tensorRow0Column1 = tensorRow0[1]
        
        XCTAssertEqual(tensorRow0Column0.shape, [2])
        XCTAssertEqual(tensorRow0Column0.rank1Data, data4d[0][0][0])
        XCTAssertEqual(tensorRow0Column1.shape, [2])
        XCTAssertEqual(tensorRow0Column1.rank1Data, data4d[0][0][1])
        
        let tensorRow0Column0Channel0 = tensorRow0Column0[0]
        let tensorRow0Column0Channel1 = tensorRow0Column0[1]
        
        XCTAssertEqual(tensorRow0Column0Channel0.shape, [])
        XCTAssertEqual(tensorRow0Column0Channel0.rank0Data, data4d[0][0][0][0])
        XCTAssertEqual(tensorRow0Column0Channel1.shape, [])
        XCTAssertEqual(tensorRow0Column0Channel1.rank0Data, data4d[0][0][0][1])
        
        let scalarTensor = Tensor(Float(5))
        XCTAssertEqual(scalarTensor[2].shape, [])
        XCTAssertEqual(scalarTensor[1].rank0Data, 5)
    }
    
    func testEquatability() {
        let tensor1 = Tensor([[Float]]([[1, 2], [3, 4]]))
        let tensor2 = Tensor([[Float]]([[1, 2], [3, 4]]))
        let tensor3 = Tensor([[Float]]([[5, 6], [7, 8]]))
        let tensor4 = Tensor([Float]([1, 2, 3, 4]))
        
        XCTAssertEqual(tensor1, tensor2)
        XCTAssertNotEqual(tensor1, tensor3)
        XCTAssertNotEqual(tensor1, tensor4)
    }
    
    func testIsEqualForFloat() {
        let accuracy: Float = 0.0001
        let tensor = Tensor([[[Float]]]([[[1, 2], [3,4]], [[5, 6], [7, 8]]]))
        let tensorWithOtherShape = Tensor([Float]([1, 2, 3]))
        let tensorWithinAccuracy = Tensor([[[Float]]]([
            [[1, 2 + accuracy / 2], [3,4]],
            [[5 + accuracy / 2, 6], [7, 8]]]))
        let tensorOutOfAccuracy = Tensor([[[Float]]]([
            [[1 + 2 * accuracy, 2], [3,4]],
            [[5, 6 + 2 * accuracy], [7 , 8]]]))

        XCTAssertFalse(tensor.isEqual(tensorWithOtherShape, accuracy: accuracy),
        "The tensor shapes should match.")
        XCTAssertTrue(tensor.isEqual(tensorWithinAccuracy, accuracy: accuracy),
        "\(tensor) is not equal \(tensorWithinAccuracy)")
        XCTAssertFalse(tensor.isEqual(tensorOutOfAccuracy, accuracy: accuracy),
        "\(tensor) is equal \(tensorOutOfAccuracy)")
        XCTAssertTrue(Tensor(shape: [4]).isEqual(Tensor(shape: [4]), accuracy: accuracy))
        XCTAssertFalse(Tensor(shape: [4]).isEqual(Tensor([Float]([1,2,3,4])), accuracy: accuracy))
    }
}

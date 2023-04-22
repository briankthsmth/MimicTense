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

final class TensorCombiningTests: XCTestCase {
    let data = TensorTestData()
    
    func testAppendVectorTensors() {
        var baseTensor = Tensor(shape: [], dataType: .float32)
        let vectorTensor1 = Tensor([Float]([1, 2, 3]))
        baseTensor.append(vectorTensor1)
        XCTAssertEqual(baseTensor.shape, [1,3])
        XCTAssertEqual(baseTensor.data, vectorTensor1.data)
        
        let vectorTensor2 = Tensor([Float]([4, 5, 6]))
        baseTensor.append(vectorTensor2)
        XCTAssertEqual(baseTensor.shape, [2,3])
        XCTAssertEqual(baseTensor.data, vectorTensor1.data + vectorTensor2.data)
    }
    
    func testAppendRank4Tensors() {
        var baseTensor = Tensor(shape: [],
                                dataType: data.rank4TensorFeatureChannelFirst.dataType)
        baseTensor.append(data.rank4TensorFeatureChannelFirst)
        XCTAssertEqual(baseTensor, data.rank4TensorFeatureChannelFirst)
        
        let originalBatchSize = baseTensor.shape[0]
        baseTensor.append(data.rank4TensorFeatureChannelFirst)
        XCTAssertEqual(baseTensor.shape[0], originalBatchSize + data.rank4TensorFeatureChannelFirst.shape[0])
        XCTAssertEqual(baseTensor.data, data.rank4TensorFeatureChannelFirst.data + data.rank4TensorFeatureChannelFirst.data)
    }
}

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
//  Created by Brian Smith on 6/15/22.
//

import XCTest
import MimicTransferables

public func assertEqual(resultTensor: Tensor, expectedVector: [Float]) {
    var resultVector = [Float](repeating: 0, count: resultTensor.shape.reduce(1, *))
    let _ = resultVector.withUnsafeMutableBytes { pointer in
        resultTensor.data.copyBytes(to: pointer)
    }
    zip(resultVector, expectedVector).forEach {
        XCTAssertEqual($0.0, $0.1, accuracy: 0.0001)
    }
}

public func assertEqual<T>(_ tensor1: Tensor, _ tensor2: Tensor, accuracy: T) throws where T: FloatingPoint {
    let dataType =  try XCTUnwrap(DataType(T.self), "Unsupported native data type for accuracy.")
    XCTAssertEqual(tensor1.dataType, dataType)
    XCTAssertEqual(tensor2.dataType, dataType)
    try XCTSkipUnless(tensor1.dataType == tensor2.dataType, "Data types not the same.")
    XCTAssertEqual(tensor1.shape, tensor2.shape)
    try XCTSkipUnless(tensor1.shape == tensor2.shape, "The shapes do not match.")
    
    tensor1.data.withUnsafeBufferPointer { pointer1 in
        pointer1.withMemoryRebound(to: T.self) { buffer1 in
            tensor2.data.withUnsafeBufferPointer { pointer2 in
                pointer2.withMemoryRebound(to: T.self) { buffer2 in
                    zip(buffer1, buffer2).forEach {
                        XCTAssertEqual($0.0, $0.1, accuracy: accuracy)
                    }
                }
            }
        }
    }
}

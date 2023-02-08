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
//  Created by Brian Smith on 5/23/22.
//

import XCTest
import MimicTransferables

class TensorTests: XCTestCase {
    let data = TensorTestData()
    
    func testScalarValue() throws {
        // test data conversion
        XCTAssertEqual(Tensor(data.scalar).extractScalar(), data.scalar)
    }
    
    func testFeatureChannelCount() throws {
        // Test scalar
        XCTAssertEqual(Tensor(shape: [], dataType: .float32).featureChannelCount, 0,
                       "Scalar should not have a channel count.")
        
        // Test rank 4 tensor
        XCTAssertEqual(data.rank4TensorFeatureChannelFirst.featureChannelCount,
                       data.featureChannelFirstArray[0].count)
        XCTAssertEqual(data.rank4TensorFeatureChannelLast.featureChannelCount,
                       data.featureChannelLastArray[0][0][0].count)
    }
    
    func testDefaultFeatureChannelPosition() {
        XCTAssertEqual(Tensor(shape: [], dataType: .float32).featureChannelPosition, .notApplicable)
        XCTAssertEqual(Tensor(shape: [1], dataType: .float32).featureChannelPosition, .notApplicable)
        XCTAssertEqual(Tensor(shape: [1, 3], dataType: .float32).featureChannelPosition, .notApplicable)
        XCTAssertEqual(Tensor(shape: [1, 3, 4], dataType: .float32).featureChannelPosition, .notApplicable)
        XCTAssertEqual(Tensor(shape: [1, 2, 3, 4], dataType: .float32).featureChannelPosition, .last)
    }
    
    func testShapeCount() {
        let tensor = Tensor(shape: [3, 2], dataType: .float32)
        XCTAssertEqual(tensor.shapeCount, 3 * 2)
    }
    
    func testShapeByteCount() {
        let tensor = Tensor(shape: [4, 5], dataType: .float32)
        XCTAssertEqual(tensor.shapeByteCount, 4 * 5 * MemoryLayout<Float>.size)
    }
}

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
//  Created by Brian Smith on 5/19/23.
//

import XCTest
import MetalPerformanceShadersGraph
import MimicTransferables
@testable import MpsgComputeEngineModule

final class TensorMpsgConversionsTests: XCTestCase {
    struct Constant {
        static let shape = [24, 24, 3]
        static let dataType = DataType.float32
    }
    
    func testPlaceholderTensorConversion() {
        let graph = MPSGraph()
        let tensor = Tensor(shape: Constant.shape, dataType: Constant.dataType)
        let platformTensor = tensor.makeMpsgTensor(for: graph)
        XCTAssertEqual(platformTensor.shape, Constant.shape.map { NSNumber(value: $0) })
    }
    
    func testDataTensorConversion() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let swiftRange = Float(0)...Float(1)
        let randomDesciptor = RandomDescriptor(type: .uniformNow, range: Range(swiftRange))
        let tensor = Tensor(shape: Constant.shape, dataType: Constant.dataType, randomDescriptor: randomDesciptor)
        let platformTensor = tensor.makeMpsgTensorData(for: MPSGraphDevice(mtlDevice: device))
        XCTAssertEqual(platformTensor.shape, tensor.shape.map { NSNumber(value: $0) })
    }
}

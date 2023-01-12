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
//
//  Created by Brian Smith on 10/4/22.
//

import XCTest
import MLCompute
@testable import MimicComputeEngine
import MimicTransferables

final class MlcTensorConversionsTests: XCTestCase {
    let vector: [Float] = [1, 2, 3, 4]
    var tensorData: MLCTensorData!
    var expectedDataBuffer: [UInt8]!
    var tensor: MLCTensor!
    var tensorBoundToDevice: MLCTensor!
    
    override func setUpWithError() throws {
        tensorData = MLCTensorData(immutableBytesNoCopy: vector,
                                   length: vector.count * MemoryLayout<Float>.size)
        
        let vectorByteCount = vector.count * MemoryLayout<Float>.size
        expectedDataBuffer = [UInt8](unsafeUninitializedCapacity: vectorByteCount) { buffer, initializedCount in
            vector.withUnsafeBufferPointer { pointer in
                pointer.withMemoryRebound(to: UInt8.self) { vectorBuffer in
                    vectorBuffer.copyBytes(to: buffer, count: vectorByteCount)
                    initializedCount = vectorByteCount
                }
            }
        }
        
        tensor = MLCTensor(shape: [vector.count], data: tensorData, dataType: .float32)
        let device = try XCTUnwrap(MLCDevice(type: .gpu))
        tensorBoundToDevice = MLCTensor(shape: [vector.count], dataType: .float32)
        tensorBoundToDevice.bindAndWriteData(tensorData, to: device)
    }
    
    func testCopyDataToArray() throws {
        XCTAssertEqual(tensor.copyToArray(), vector)
    }
    
    func testCopyDataFromDeviceToArray() throws {
        XCTAssertEqual(tensorBoundToDevice.copyToArray(), vector)
    }
    
    func testCopyDataToBuffer() throws {
        XCTAssertEqual(tensor.copyDataToBuffer(), expectedDataBuffer)
    }
    
    func testCoDataFromDeviceToBuffer() throws {
        XCTAssertEqual(tensorBoundToDevice.copyDataToBuffer(), expectedDataBuffer)
    }
    
    func testMakeTransferableTensor() throws {
        let transferableTensor = tensor.makeTensor()
        XCTAssertEqual(transferableTensor.shape, tensor.descriptor.shape)
        XCTAssertEqual(transferableTensor.data, tensor.copyDataToBuffer())
        XCTAssertEqual(transferableTensor.dataType, DataType(tensor.descriptor.dataType))
        XCTAssertEqual(transferableTensor.featureChannelPosition, .notApplicable)
        XCTAssertEqual(transferableTensor.randomInitializerType, .none)
    }
}

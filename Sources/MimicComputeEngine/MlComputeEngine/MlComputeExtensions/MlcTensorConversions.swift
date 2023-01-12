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

import Foundation
import MLCompute
import MimicCore
import MimicTransferables

extension MLCTensor {
    ///  Factory method to create a Tensor.
    ///
    ///  - Returns: A Tensor created from the MLCTensor.
    func makeTensor() -> Tensor {
        return Tensor(shape: descriptor.shape,
                      data: copyDataToBuffer(),
                      dataType: .float32)
    }
    
    /// Copy the tensor's data to an array.
    ///
    /// Copies the data only if the generic type is the same as the tensors data type.
    ///
    /// - Returns: An array with the data.
    func copyToArray<NativeType: PlatformSupportedNativeType>() -> [NativeType] {
        guard descriptor.dataType == NativeType.platformDataType else { return [] }
        let dataBuffer = copyDataToBuffer()
        let length = dataBuffer.count / NativeType.memoryLayoutSize
        let array = [NativeType](unsafeUninitializedCapacity: length) { arrayBuffer, initializedCount in
            arrayBuffer.withMemoryRebound(to: UInt8.self) { buffer in
                let _ = dataBuffer.copyBytes(to: buffer, count: dataBuffer.count)
            }
            initializedCount = length
        }
        return array
    }
    
    /// Copy the tensor data to a UInt8 array.
    /// - Returns: An UInt8 array containing the data associated with the tensor.
    func copyDataToBuffer() -> [UInt8] {
        guard
            let shapeFirstElement = descriptor.shape.first,
            let strideFirstElement = descriptor.stride.first
        else {
            return []
        }
        let byteCount = shapeFirstElement * strideFirstElement
        return [UInt8](unsafeUninitializedCapacity: byteCount) { buffer, initializedCount in
            guard let bufferAddress = buffer.baseAddress else {
                initializedCount = 0
                return
            }
            if let _ = device {
                copyDataFromDeviceMemory(toBytes: bufferAddress,
                                         length: byteCount,
                                         synchronizeWithDevice: false)
            } else {
                guard
                    let data = data,
                    data.count == byteCount
                else {
                    initializedCount = 0
                    return
                }
                data.copyBytes(to: bufferAddress, count: byteCount)
            }
            initializedCount = byteCount
        }
    }
}

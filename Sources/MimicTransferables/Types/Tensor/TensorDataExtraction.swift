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

import Foundation
import MimicCore

extension Tensor {
    public func extractScalar<NativeType: SupportedNativeType>(_ nativeType: NativeType.Type = NativeType.self) -> NativeType? {
        guard
            DataType(nativeType) == dataType,
            data.count == MemoryLayout<NativeType>.size,
            shape.isEmpty
        else {
            return nil
        }
        return data.withUnsafeBytes { $0.load(as: nativeType) }
    }
    
    public func extract<NativeType: SupportedNativeType>(_ nativeType: NativeType.Type = NativeType.self) -> Any? {
        let dataCount = shape.reduce(1, *) * MemoryLayout<NativeType>.size
        guard
            DataType(nativeType) == dataType,
            data.count == dataCount,
            dataCount > 0
        else {
            return nil
        }
        
        var vector = [NativeType](repeating: NativeType(), count: dataCount)
        let _ = vector.withUnsafeMutableBytes { pointer in
            data.copyBytes(to: pointer, count: dataCount)
        }
        
        if shape.isEmpty { return vector[0] }
        
        var multiArray: Array<Any> = vector
        shape
            .reversed()
            .forEach { count in
                multiArray = multiArray
                    .enumerated()
                    .reduce(into: [[Any]]()) { partialResult, tuple in
                        let index = tuple.offset / count
                        var subArray = partialResult.count > index ? partialResult.remove(at: index) : []
                        subArray.append(tuple.element)
                        partialResult.append(subArray)
                    }
            }
        return multiArray[0]
    }
}

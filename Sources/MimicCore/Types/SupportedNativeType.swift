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


/// Protocol for native types that are supported in graphs.
public protocol SupportedNativeType: Random {
    /// The zero value for the type.
    static var zero: Self { get }
    /// The type's size in number of bytes.
    static var memoryLayoutSize: Int { get }
    
    init()
    
    /// Factory to create a byte buffer representation for the type.
    func makeBuffer() -> [UInt8]
    /// Factory to create a type's value from a buffer
    static func makeValue(from buffer: [UInt8]) throws -> Self
}


extension Float: SupportedNativeType {
    enum NativeTypeError: Error {
        case incompatibleBufferSize
    }
    
    public static var zero: Float { 0 }
    public static var memoryLayoutSize: Int { MemoryLayout<Self>.size }
    
    public func makeBuffer() -> [UInt8] {
        return [UInt8](unsafeUninitializedCapacity: Self.memoryLayoutSize) { buffer, initializedCount in
            initializedCount = withUnsafeBytes(of: self) { $0.bindMemory(to: UInt8.self).copyBytes(to: buffer) }
        }
    }
    
    static public func makeValue(from buffer: [UInt8]) throws -> Float {
        guard buffer.count == Self.memoryLayoutSize else {
            throw NativeTypeError.incompatibleBufferSize
        }
        
        var value: Float = 0
        withUnsafeMutableBytes(of: &value) {
            let _ = buffer.copyBytes(to: $0)
        }
        return value
    }
}

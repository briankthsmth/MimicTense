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
//  Created by Brian Smith on 5/26/23.
//

import Foundation
import MimicCore

/// A descriptor to use to generate random distributions.
public struct RandomDescriptor: Transferable, Equatable {
    /// The type of random distribution to use.
    public let type: RandomInitializerType
    /// The range to use to create random numbers with.
    public let range: Range
    
    /// Initializes the descriptor.
    ///
    /// - Parameters:
    ///   - type: The type of random distribution to initialize data.
    ///   - range: The range for the random distribution.
    ///
    public init(type: RandomInitializerType, range: Range) {
        self.type = type
        self.range = range
    }
    
    /// Initilizes the descriptor with a Swift Range.
    ///
    /// - Parameter:
    ///   - type: The type of random distribution to initialize data.
    ///   - range: A Swift Range for the random distribution.
    ///
    public init<NativeType: SupportedNativeType>(type: RandomInitializerType, range: Swift.Range<NativeType>) {
        self.init(type: type, range: Range(range))
    }
    
    /// Initilizes the descriptor with a Swift ClosedRange.
    ///
    /// - Parameter:
    ///   - type: The type of random distribution to initialize data.
    ///   - range: A Swift ClosedRange for the random distribution.
    ///
    public init<NativeType: SupportedNativeType>(type: RandomInitializerType, range: ClosedRange<NativeType>) {
        self.init(type: type, range: Range(range))
    }
}

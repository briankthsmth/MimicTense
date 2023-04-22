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
    public init<NativeType: SupportedNativeType>(_ value: NativeType) {
        self.init(shape: [], value: [value])
    }
    
    public init<NativeType: SupportedNativeType>(_ value: [NativeType]) {
        self.init(shape: [value.count], value: value)
    }
    
    public init<NativeType: SupportedNativeType>(_ value: [[NativeType]]) {
        let shape = [value.count,
                     value.first?.count ?? 0]
        self.init(shape: shape, value: Array(value.joined()))
    }
    
    public init<NativeType: SupportedNativeType>(_ value: [[[NativeType]]]) {
        let shape = [value.count,
                     value.first?.count ?? 0,
                     value.first?.first?.count ?? 0]
        self.init(shape: shape, value: Array(value.joined().joined()))
    }
    
    public init<NativeType: SupportedNativeType>(_ value: [[[[NativeType]]]],
                                                 featureChannelPosition: FeatureChannelPosition = .last)
    {
        let shape = [value.count,
                     value.first?.count ?? 0,
                     value.first?.first?.count ?? 0,
                     value.first?.first?.first?.count ?? 0]
        self.init(shape: shape,
                  value: Array(value.joined().joined().joined()),
                  featureChannelPosition: featureChannelPosition)
    }
    
    /// Copy initializer that allows the for changing property values.
    public init(_ original: Tensor,
                shape: [Int]? = nil,
                data: [UInt8]? = nil,
                dataType: DataType? = nil,
                featureChannelPosition: FeatureChannelPosition? = nil,
                randomInitializerType: RandomInitializerType? = nil)
    {
        self.shape = shape ?? original.shape
        self.data = data ?? original.data
        self.dataType = dataType ?? original.dataType
        self.featureChannelPosition = featureChannelPosition ?? original.featureChannelPosition
        self.randomInitializerType = randomInitializerType ?? original.randomInitializerType
    }
    
    private init<NativeType: SupportedNativeType>(shape: [Int],
                                                  value: [NativeType],
                                                  featureChannelPosition: FeatureChannelPosition = .last )
    {
        self.init(shape: shape,
                  data: value.makeBuffer(),
                  dataType: DataType(NativeType.self),
                  featureChannelPosition: featureChannelPosition)
    } 
}

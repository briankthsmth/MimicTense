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
//  Created by Brian Smith on 7/20/22.
//

import Foundation
import MimicTransferables
import MimicCore

public struct FullyConnected<NativeType: NeuralNativeType>: Layerable {
    public let identifier: LayerIdentifier = LayerIdentifier(kind: .fullyConnected)
    public var name: String?
    
    public let inputs: Inputs<NativeType>?
    public let weights: Tensor<NativeType>?
    public let biases: Tensor<NativeType>?
    public let inputFeatureChannelCount: Int?
    public let outputFeatureChannelCount: Int?
    
    public let arithmeticOperation: MimicCore.ArithmeticOperation? = nil
    public let kernelSize: KernelSize? = nil
    
    public init(name: String? = nil,
                weights: Tensor<NativeType>,
                biases: Tensor<NativeType>? = nil,
                inputFeatureChannelCount: Int,
                outputFeatureChannelCount: Int,
                _ makeInputs: (() -> Inputs<NativeType>)? = nil)
    {
        inputs = makeInputs?()
        
        self.name = name
        self.weights = weights
        self.biases = biases
        self.inputFeatureChannelCount = inputFeatureChannelCount
        self.outputFeatureChannelCount = outputFeatureChannelCount
    }
}

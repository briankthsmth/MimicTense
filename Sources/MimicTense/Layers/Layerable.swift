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
//  Created by Brian Smith on 4/19/22.
//

import Foundation
import MimicTransferables
import MimicCore

public protocol Layerable {
    associatedtype NativeType: NeuralNativeType
    
    var identifier: LayerIdentifier { get }
    
    var inputs: Inputs<NativeType>? { get }
    var weights: Tensor<NativeType>? { get }
    var biases: Tensor<NativeType>? { get }
    
    var arithmeticOperation: ArithmeticOperation? { get }
    
    var kernelSize: KernelSize? { get }
    var inputFeatureChannelCount: Int? { get }
    var outputFeatureChannelCount: Int? { get }
}

extension Layerable {
    func makeTransferableInputTensors() throws -> [MimicTransferables.Tensor]? {
        guard
            let tensors = inputs?.tensors,
            !tensors.isEmpty
        else {
            return nil
        }
        
        return try tensors.map { try $0.makeTransferable() }
    }
    
    func makeTransferable() throws -> Layer {
        Layer(kind: identifier.kind,
              dataType: DataType(NativeType.self),
              arithmeticOperation: arithmeticOperation,
              kernelSize: try kernelSize?.makeTransferable(),
              inputFeatureChannelCount: inputFeatureChannelCount,
              outputFeatureChannelCount: outputFeatureChannelCount,
              weights: try weights?.makeTransferable(),
              biases: try biases?.makeTransferable()
        )
    }
}


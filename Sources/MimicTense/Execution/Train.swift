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
//  Created by Brian Smith on 2/8/23.
//

import Foundation
import MimicTransferables

/// Class to build a training operation.
public class Train<NativeType: NeuralNativeType>: Compilable, Executable {
    /// Property to an output stream for tensor results from each batch.
    public var outputStream: AsyncThrowingStream<Tensor<NativeType>, Error> {
        return sessionRunner.makeOutputStream()
    }
    
    /// Initializer that uses a result builder closure to build an training operation from a data set and graphs.
    ///
    ///  - Parameters:
    ///    - epochs: Then number of epochs to perform.
    ///    - lossFunction: The loss function.
    ///    - optimizer: The type of optimizer to use.
    ///    - make: Result builder closure that builds a data set and graphs.
    public init(epochs: Int,
                lossFunction: LossFunctionType,
                optimizer: OptimizerType,
                @ExecutionGraphBuilder _ make: () -> (dataSet: any MimicTense.DataBatchable, graph: any Graphable)) throws
    {
        let input = make()
        sessionRunner = try SessionRunner(kind: .training(lossFunction: lossFunction,
                                                          optimizer: optimizer),
                                          epochs: epochs,
                                          dataSet: input.dataSet,
                                          graph: input.graph)
    }
    
    /// Compiles the graph on to the given device for training.
    ///
    /// - Parameters:
    ///   - device: The device type.
    ///
    /// - Returns: A reference to the Train object.
    public func compile(device: Device) async throws -> Self {
        try await sessionRunner.compile(device: device)
        return self
    }
    
    /// Retrieve a tensor with the trained weights for a layer.
    ///
    ///  - Parameters:
    ///    - layer: The name for a layer in the model.
    ///
    ///  - Returns: A tensor with the traiined weights.
    public func retrieveWeights(for layer: String) throws -> Tensor<NativeType> {
        let layer = try sessionRunner.retrieveLayer(for: layer)
        guard let weights = layer.weights else {
            throw LayerError.unsetWeights
        }
        return try Tensor<NativeType>.make(from: weights)
    }
    
    /// Retrieve a tensor with the trained weights for a layer.
    ///
    ///  - Parameters:
    ///    - layer: The name for a layer in the model.
    ///
    ///  - Returns: A tensor with the traiined weights.
    public func retrieveBiases(for layer: String) throws -> Tensor<NativeType> {
        let layer = try sessionRunner.retrieveLayer(for: layer)
        guard let biases = layer.biases else {
            throw LayerError.unsetBiases
        }
        return try Tensor<NativeType>.make(from: biases)
    }
        
    // MARK: Private Interface
    private var sessionRunner: SessionRunner<NativeType>
}

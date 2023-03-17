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
//  Created by Brian Smith on 9/27/22.
//

import Foundation
import MimicTransferables
import MLCompute

/// Class to  train a set of neural network graph.
///
final class MlComputeTrainingGraph:
    TrainingGraphable,
    PlatformExecutionGraphable,
    ModelInspectable
{
    /// Initializer for training graph.
    ///
    ///  - Parameters:
    ///     - graph: A graph to use to create the training graph.
    ///     - lossLabelTensor: Placeholder tensor with the shape of the loss labels.
    ///     - lossFunction: The loss function used to calculate the loss errors.
    ///     - optimizer: The optimizer type to use.
    init(graph: Graph,
         lossLabelTensor: Tensor,
         lossFunction: LossFunctionType,
         optimizer: OptimizerType) throws
    {
        self.graph = graph
        self.lossFunction = lossFunction
        self.optimizer = optimizer
        
        let product = try Self.makePlatformGraph(from: graph)
        platformGraph = product.graph
        outputTensor = product.output
        platformTrainingGraph = MLCTrainingGraph(graphObjects: [product.graph],
                                                 lossLayer: lossFunction.makeMlcLossLayer(),
                                                 optimizer: optimizer.makeMlcOptimizer())
        platformTrainingGraph.addInputs(product.inputs.makeInputDictionary(startingWith: Constant.inputPrefix),
                                        lossLabels: [lossLabelTensor.makeMlcTensor()].makeInputDictionary(startingWith: Constant.lossLabelPrefix))
    }
    
    /// Compile the graph to a paticular device
    ///
    /// - Parameters:
    ///    - device: The device type to compile the graphs onto.
    func compile(device: DeviceType) throws {
        guard let mlcDevice = MLCDevice(type: device.mlcDeviceType) else {
            throw ComputeEngineError.deviceNotAvailable
        }
        platformTrainingGraph.compile(device: mlcDevice)
    }
    
    /// Execute a training run for a single batch of data.
    ///
    /// - Parameters:
    ///    - inputs: Tensors with data for each input in the graph.
    ///    - lossLables: Tensors with label data for the batch.
    ///    - batchSize: The size of the batchs in the tensor data.
    ///
    /// - Returns: Output tensors for the batch.
    func execute(inputs: [Tensor], lossLables: Tensor, batchSize: Int) async throws -> Tensor {
        return try await withCheckedThrowingContinuation{ continuation in
            let inputsData = inputs
                .map { $0.makeMlcTensorData() }
                .makeInputDictionary(startingWith: Constant.inputPrefix)
            let lossLabelsData = [lossLables.makeMlcTensorData()]
                .makeInputDictionary(startingWith: Constant.lossLabelPrefix)
            
            platformTrainingGraph.execute(inputsData: inputsData,
                                          lossLabelsData: lossLabelsData,
                                          lossLabelWeightsData: nil,
                                          batchSize: batchSize) { tensor, error, _ in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let tensor = tensor else {
                    continuation.resume(throwing: ComputeEngineError.invalidOutput)
                    return
                }
                continuation.resume(returning: tensor.makeTensor())
            }
        }
    }
    
    /// Retrieves all the graphs and their layers from device memory.
    ///
    /// Retrieves all of the graphs along with the  data that may have been updated during
    ///  training.
    /// - Returns: All of the graphs for the model used in training.
    func retrieveGraph() throws -> Graph {
        platformTrainingGraph.synchronizeUpdates()
        return try platformGraph.makeGraph(against: graph)
    }
    
    /// Retrieve the layer from the device memory identified by the given label.
    ///
    /// This method retrieves a layer by name and it will pull any data from device memory  that was
    /// updated during training for the layer. This method will throw an exception if the layer
    /// does not exist for the given label.
    ///
    /// - Parameters:
    /// - label: A string that identifies the layer.
    ///
    /// - Returns: The layer corrisponding to the label.
    func retrieveLayer(by label: String) throws -> Layer {
        platformTrainingGraph.synchronizeUpdates()
        guard
            let originalLayer = graph.layers.first(where: { $0.label == label }),
            let platformLayer = platformTrainingGraph.layers.first(where: { $0.label.contains(label) })
        else {
            throw ComputeEngineError.layerConversion
        }
        return try platformLayer.makeLayer(against: originalLayer)
    }
    
    // Mark: Private Interface
    private struct Constant {
        static let inputPrefix = "input"
        static let lossLabelPrefix = "lossLabel"
    }
    
    private let graph: Graph
    private let lossFunction: LossFunctionType
    private let optimizer: OptimizerType
    
    private let platformGraph: MLCGraph
    private let platformTrainingGraph: MLCTrainingGraph
    private let outputTensor: MLCTensor
}

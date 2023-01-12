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
final class MlComputeTrainingGraph: TrainingGraphable, PlateformExecutionGraphable {
    /// Initializer for training graph.
    ///
    ///  - Parameters:
    ///     - graphs: An array of graphs to use to create the training graph.
    ///     - lossLabelTensor: Placeholder tensors with the shape of the loss labels for each graph.
    ///     - lossFunction: The loss function used to calculate the loss errors.
    ///     - optimizer: The optimizer type to use.
    init(graphs: [Graph],
         lossLabelTensors: [Tensor],
         lossFunction: LossFunctionType,
         optimizer: OptimizerType) throws
    {
        self.graphs = graphs
        self.lossFunction = lossFunction
        self.optimizer = optimizer
        
        let platformGraphs = try Self.makePlatformGraphs(from: graphs)
        outputTensors = platformGraphs.outputs
        platformTrainingGraph = MLCTrainingGraph(graphObjects: platformGraphs.graphs,
                                                 lossLayer: lossFunction.makeMlcLossLayer(),
                                                 optimizer: optimizer.makeMlcOptimizer())
        platformTrainingGraph.addInputs(platformGraphs.inputs.makeInputDictionary(startingWith: Constant.inputPrefix),
                                        lossLabels: lossLabelTensors.map { $0.makeMlcTensor() }.makeInputDictionary(startingWith: Constant.lossLabelPrefix))
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
    ///    - inputs: Tensors with data for each input in the graphs.
    ///    - lossLables: Tensors with data for each graph.
    ///    - batchSize: The size of the batchs in the tensor data.
    func execute(inputs: [Tensor], lossLables: [Tensor], batchSize: Int) async throws {
        let _: Void = try await withCheckedThrowingContinuation{ continuation in
            let inputsData = inputs
                .map { $0.makeMlcTensorData() }
                .makeInputDictionary(startingWith: Constant.inputPrefix)
            let lossLabelsData = lossLables
                .map { $0.makeMlcTensorData() }
                .makeInputDictionary(startingWith: Constant.lossLabelPrefix)
            
            platformTrainingGraph.execute(inputsData: inputsData,
                                          lossLabelsData: lossLabelsData,
                                          lossLabelWeightsData: nil,
                                          batchSize: batchSize) { _, error, _ in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume()
            }
        }
    }
    
    func copyWeights(for layer: Layer) throws -> Tensor {
        platformTrainingGraph.synchronizeUpdates()
        guard
            let layerLabel = layer.label,
            let platformLayer = platformTrainingGraph.layers.first(where: { $0.label.contains(layerLabel) })
        else {
            throw ComputeEngineError.layerConversion
        }
        switch platformLayer {
        case let fullyConnectedLayer as MLCFullyConnectedLayer:
            let weights = fullyConnectedLayer.weights.makeTensor()
            let reshape = Array(weights.shape[1 ..< weights.shape.count])
            let reshapedWeights = Tensor(weights, shape: reshape)
            return reshapedWeights
        default:
            return Tensor(shape: [], dataType: .float32)
        }
    }
    
    // Mark: Private Interface
    private struct Constant {
        static let inputPrefix = "input"
        static let lossLabelPrefix = "lossLabel"
    }
    
    private let graphs: [Graph]
    private let lossFunction: LossFunctionType
    private let optimizer: OptimizerType
    
    private let platformTrainingGraph: MLCTrainingGraph
    private let outputTensors: [MLCTensor]
}

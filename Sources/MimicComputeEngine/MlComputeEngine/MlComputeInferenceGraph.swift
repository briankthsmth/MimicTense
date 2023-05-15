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
//  Created by Brian Smith on 5/4/22.
//

import Foundation
import MLCompute
import MimicTransferables

/// Class to perform inference on a set of neural network graphs.
/// 
final class MlComputeInferenceGraph:
    InferenceGraphable,
    ModelInspectable
{
    init(graph: Graph) throws {
        self.graph = graph
        
        let converted = try graph.makePlatformGraph()
        self.outputTensor = converted.output
        platformInferencGraph = MLCInferenceGraph(graphObjects: [converted.graph])
        platformInferencGraph.addInputs(converted.inputs.makeInputDictionary(startingWith: Constant.inputPrefix))
    }
    
    func compile(device: DeviceType) throws {
        guard platformInferencGraph.layers.count > 0 else { return }
        platformInferencGraph.compile(device: MLCDevice(type: device.mlcDeviceType)!)
    }
    
    /// Excute an inference run on the graph.
    ///
    ///  - Parameters:
    ///    - inputs: The tensors with batch data for each input.
    ///    - batchSize:  The number of  input data in each batch.
    ///
    ///   - Returns: A tensor with the batched output data.
    func execute(inputs: [Tensor], batchSize: Int) async throws -> Tensor {
        return await withCheckedContinuation { continuation in
            let inputsData = inputs
                .map { $0.makeMlcTensorData() }
                .makeInputDictionary(startingWith: Constant.inputPrefix)
            
            platformInferencGraph.execute(inputsData: inputsData, batchSize: batchSize) { tensor, error, time in
                continuation.resume(returning: self.retrieveOutput())
            }
        }
    }
        
    func retrieveOutput() -> Tensor {
        convert(from: outputTensor)
    }
        
    func retrieveGraph() throws -> Graph {
        return graph
    }
    
    // MARK: Private Interface
    private struct Constant {
        static let inputPrefix = "input"
    }
    
    private let graph: Graph
    private let platformInferencGraph: MLCInferenceGraph
    private let outputTensor: MLCTensor
    
    private func convert(from platformTensor: MLCTensor) -> Tensor {
        let dataType = DataType(platformTensor.descriptor.dataType) ?? .float32
        let length = platformTensor.descriptor.shape.reduce(1, *) * dataType.memoryLayoutSize
        let data = [UInt8](unsafeUninitializedCapacity: length) { buffer, initializedCount in
            platformTensor.copyDataFromDeviceMemory(toBytes: buffer.baseAddress!,
                                                    length: length,
                                                    synchronizeWithDevice: false)
            initializedCount = length
        }
        return Tensor(shape: platformTensor.descriptor.shape,
                      data: data,
                      dataType: dataType)
    }
}



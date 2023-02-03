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
final class MlComputeInferenceGraph: InferenceGraphable, PlatformExecutionGraphable {    
    init(graphs: [Graph]) throws {
        self.graphs = graphs
        
        let converted = try Self.makePlatformGraphs(from: graphs)
        self.outputTensors = converted.outputs
        platformInferencGraph = MLCInferenceGraph(graphObjects: converted.graphs)
        platformInferencGraph.addInputs(converted.inputs.makeInputDictionary(startingWith: Constant.inputPrefix))
    }
    
    func compile(device: DeviceType) {
        guard platformInferencGraph.layers.count > 0 else { return }
        platformInferencGraph.compile(device: MLCDevice(type: device.mlcDeviceType)!)
    }
    
    func execute(inputs: [Tensor], batchSize: Int) async -> [Tensor] {
        return await withCheckedContinuation { continuation in
            let inputsData = inputs
                .map { $0.makeMlcTensorData() }
                .makeInputDictionary(startingWith: Constant.inputPrefix)
            
            platformInferencGraph.execute(inputsData: inputsData, batchSize: batchSize) { tensor, error, time in
                continuation.resume(returning: self.retrieveOutputs())
            }
        }
    }
        
    func retrieveOutputs() -> [Tensor] {
        outputTensors.map { convert(from: $0) }
    }
    
    func retrieveOutputTensor(at index: Int) -> Tensor {
        convert(from: outputTensors[index])
    }
    
    // MARK: Private Interface
    private struct Constant {
        static let inputPrefix = "input"
    }
    
    private let graphs: [Graph]
    private let platformInferencGraph: MLCInferenceGraph
    private let outputTensors: [MLCTensor]
    
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



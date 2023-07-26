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
//  Created by Brian Smith on 7/25/23.
//

import Foundation
import MetalPerformanceShadersGraph

import MimicTransferables
@testable import MpsgComputeEngineModule

struct MpsgGraphTestRunner {
    enum ClassError: Error {
        case mtlDevice
        case commandQueue
        case outputTensorDataMissing
    }
    
    let device: MTLDevice
    let graph: MPSGraph
    
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw ClassError.mtlDevice }
        self.device = device
        graph = MPSGraph()
    }
    
    func run(inputs: [Tensor], layerFactory: ([MPSGraphTensor], MTLDevice, MPSGraph) throws -> (output: MPSGraphTensor, weightsPair: Layer.TensorPair?)) throws -> Tensor {
        let graphDevice = MPSGraphDevice(mtlDevice: device)
        
        let mpsgInputs = inputs.map { $0.makeMpsgTensor(for: graph) }
        let tensors = try layerFactory(mpsgInputs, device, graph)
        
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        try inputs.enumerated().forEach {
            feeds[mpsgInputs[$0.offset]] = try $0.element.makeMpsgTensorData(for: graphDevice)
        }
        if let weightsPair = tensors.weightsPair {
            feeds[weightsPair.placeholder] = weightsPair.data
        }
        
        
        guard let commandQueue = device.makeCommandQueue() else { throw ClassError.commandQueue }
        let result = graph.run(with: commandQueue,
                               feeds: feeds,
                               targetTensors: [tensors.output],
                               targetOperations: nil)
        
        guard let tensorData = result[tensors.output] else { throw  ClassError.outputTensorDataMissing }
        
        return try tensorData.makeTensor()
    }
}

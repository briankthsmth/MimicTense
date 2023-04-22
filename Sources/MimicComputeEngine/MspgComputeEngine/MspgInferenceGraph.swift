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
//  Created by Brian Smith on 4/13/23.
//

import Foundation
import Metal
import MimicTransferables

final class MpsgInferenceGraph:
    InferenceGraphable,
    PlatformExecutionGraphable,
    ModelInspectable
{
    struct UnimplementedError: Error {}
    
    init(graph: Graph) throws {
        
    }
    
    func retrieveOutput() -> MimicTransferables.Tensor {
        return Tensor(shape: [], dataType: .float32)
    }
    
    func retrieveGraph() throws -> MimicTransferables.Graph {
        throw UnimplementedError()
    }
    
    func compile(device: MimicTransferables.DeviceType) throws {
        
    }
    
    func execute(inputs: [MimicTransferables.Tensor], batchSize: Int) async throws -> MimicTransferables.Tensor {
        throw UnimplementedError()
    }
    
    
}
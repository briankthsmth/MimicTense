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
//  Created by Brian Smith on 5/19/23.
//

import Foundation
import MetalPerformanceShadersGraph
import MimicTransferables
import MimicComputeEngineModule

extension Tensor {
    /// Makes a placeholder MPSGraphTensor from a placeholder Tensor object.
    ///
    /// - Parameters:
    ///   - graph: The MPSGraph that the MPSGraphTensor will be created for.
    ///
    func makeMpsgTensor(for graph: MPSGraph) -> MPSGraphTensor {
        return graph.placeholder(shape: shape.mapToMpsShape(),
                                 dataType: dataType.makeMpsDataType(),
                                 name: nil)
    }
    
    /// Makes a MPSGraphTensorData from a Tensor object.
    ///
    /// It's an error if there is no data to create the tensor.
    ///
    /// - Parameters:
    ///   - device: The MPSGraphDevice to hold the data.
    ///
    func makeMpsgTensorData(for device: MPSGraphDevice) throws -> MPSGraphTensorData {
        guard data.count == shapeByteCount else { throw ComputeEngineError.missingData }
        return MPSGraphTensorData(device: device,
                                  data: Data(data),
                                  shape: shape.mapToMpsShape(),
                                  dataType: dataType.makeMpsDataType())
    }
}
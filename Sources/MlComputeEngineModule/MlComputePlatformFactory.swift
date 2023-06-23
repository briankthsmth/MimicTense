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
//  Created by Brian Smith on 6/22/22.
//

import Foundation
import MimicTransferables
import MimicComputeEngineModule

public struct MlComputePlatformFactory: PlatformFactory {
    public init() {}
    
    public func makeInferenceGraph(graph: Graph) throws -> InferenceGraphable {
        try MlComputeInferenceGraph(graph: graph)
    }
    
    public func makeTrainingGraph(graph: Graph,
                           lossLabelTensor: Tensor,
                           lossFunction: LossFunctionType,
                           optimizer: OptimizerType) throws -> TrainingGraphable
    {
        try MlComputeTrainingGraph(graph: graph,
                                   lossLabelTensor: lossLabelTensor,
                                   lossFunction: lossFunction,
                                   optimizer: optimizer)
    }
}
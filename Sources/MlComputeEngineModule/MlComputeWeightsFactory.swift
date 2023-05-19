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
//  Created by Brian Smith on 10/5/22.
//

import Foundation
import MLCompute
import MimicTransferables
import MimicComputeEngineModule

struct MlComputeWeightsFactory {
    private let weightsTensor: Tensor
    private let weightsDescriptor: MLCTensorDescriptor
    
    init(weightsTensor: Tensor, weightsDescriptor: MLCTensorDescriptor) {
        self.weightsTensor = weightsTensor
        self.weightsDescriptor = weightsDescriptor
    }
    
    func callAsFunction() throws -> MLCTensor {
        if weightsTensor.isScalar {
            let fillData = weightsTensor.extractScalar() ?? 0
            // use tensor as fill data
            return MLCTensor(descriptor: weightsDescriptor, fillWithData: fillData)
        } else if let intializer = weightsTensor.randomInitializerType {
            return MLCTensor(descriptor: weightsDescriptor,
                             randomInitializerType: intializer.makeMlcInitializer())
        }
        
        let descriptorShapeByteCount = weightsDescriptor.shape.reduce(1, *) * weightsTensor.dataType.memoryLayoutSize
        guard
            weightsTensor.data.count == descriptorShapeByteCount
        else {
            throw ComputeEngineError.invalidWeights
        }
        
        return MLCTensor(descriptor: weightsDescriptor,
                         data: weightsTensor.makeMlcTensorData())
    }
}

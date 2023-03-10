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
//  Created by Brian Smith on 7/8/22.
//

import Foundation

public struct InferenceDataSet<NativeType: NeuralNativeType>: DataBatchable {    
    public let batchSize: Int
    public let inputs: [InputData<NativeType>]
    
    public var inputTensors: [Tensor<NativeType>] {
        inputs.map { $0.data }
    }
    
    public var labels: Tensor<NativeType>? { nil }
    
    public init(batchSize: Int, @InputDataSetBuilder<NativeType> _ make: () -> [InputData<NativeType>]) {
        self.batchSize = batchSize
        inputs = make()
    }
}

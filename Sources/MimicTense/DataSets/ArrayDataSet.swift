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
//  Created by Brian Smith on 4/20/22.
//

import Foundation

public struct ArrayDataSet<NativeType: NeuralNativeType>: DataBatchable {
    public let batchSize: Int
    public var inputTensors: [Tensor<NativeType>] {
        [tensor]
    }
    
    public var labels: Tensor<NativeType>? { nil }
    
    public init(data: [NativeType]) {
        self.batchSize = 1
        self.tensor = Tensor(data)
    }
    
    public init(data: [[NativeType]]) {
        self.batchSize = 1
        self.tensor = Tensor(data)
    }
    
    public init(data: [[[NativeType]]]) {
        self.batchSize = 1
        self.tensor = Tensor(data)
    }
    
    public init(data: [[[[NativeType]]]]) {
        self.batchSize = data.count
        self.tensor = Tensor(data)
    }
    
    private let tensor: Tensor<NativeType>
}

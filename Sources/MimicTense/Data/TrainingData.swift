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
//  Created by Brian Smith on 2/13/23.
//

import Foundation

/// Structure to contain input  data and label data for training.
public struct TrainingData<NativeType: NeuralNativeType>: DataContainable {
    public let inputs: [InputData<NativeType>]
    public let labels: LabelData<NativeType>
    
    public init(@TrainingDataBuilder<NativeType> _ make: () -> (inputs: [InputData<NativeType>],
                                                                labels: LabelData<NativeType>))
    {
        let arguments = make()
        inputs = arguments.inputs
        labels = arguments.labels
    }
}

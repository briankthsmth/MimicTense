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
//  Created by Brian Smith on 2/3/23.
//

import Foundation
import MimicTransferables

/// Protocol for execution of a training batch.
public protocol TrainingExecutable {
    /// Excute a training run on a batch of data.
    ///
    ///  - Parameters:
    ///    - inputs: Tensors for each input with the batch data.
    ///    - lossLables: Tensor with the label data for the batch.
    ///    - batchSize: Size of the batch.
    ///
    ///  - Returns: A tensor with the batch's output data.
    func execute(inputs: [Tensor], lossLables: Tensor, batchSize: Int) async throws -> Tensor
}

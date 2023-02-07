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
//  Created by Brian Smith on 1/18/23.
//

import Foundation
import MimicTransferables

/// Extension methods to handle creating batch data from datasets.
extension DataSet {
    /// A placeholder Tensor (contains no data) for a batch of labels.
    var batchLabelsPlaceholder: [Tensor]? {
        labels?.map { Tensor(shape: [batchSize] + $0.shape[1...], dataType: $0.dataType) }
    }
    
    /// Factory method to create batch input data from the data set.
    /// - Parameters:
    ///  - batch: The index for the batch to create
    ///
    /// - Returns: An array of tensors with batch input tensor data for each graph in the model.
    public func makeBatch(at batch: Int) -> [Tensor] {
        return makeBatch(with: tensors, batch: batch)
    }
    
    /// Factory method  to create a the expected label tensor for training graphs.
    ///  - Parameters:
    ///    - batch: The index for the batch to create.
    ///
    ///  - Returns: An array of tensors with batch label data for each graph in the model. Returns nil if
    ///         there are no training labels set in the dataset.
    public func makeBatchLabels(at batch: Int) -> [Tensor]? {
        guard let labels = labels else { return nil }
        return makeBatch(with: labels, batch: batch)
    }
        
    /// Internal factory method to create batches from either the input tensor or label tensor arrays.
    func makeBatch(with tensors: [Tensor], batch: Int) -> [Tensor] {
        var batchTensors = [Tensor]()
        tensors.forEach {
            batchTensors.append(makeBatch(with: $0, batch: batch))
        }
        return batchTensors
    }
    
    /// Internal factory method to create a batch from a TensorArray.
    func makeBatch(with tensor: Tensor, batch: Int) -> Tensor {
        let batchStartIndex = batch * batchSize
        let batchEndIndex = batchStartIndex + batchSize
        return tensor[batchStartIndex..<batchEndIndex]
    }
}

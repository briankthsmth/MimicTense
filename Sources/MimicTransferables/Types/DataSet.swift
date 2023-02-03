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
//  Created by Brian Smith on 5/20/22.
//

import Foundation

/// Transferable type for datasets used in inference and training.
public struct DataSet: Transferable {
    /// An array of tensors for each graph's inputs in a model.
    public let tensors: [Tensor]
    /// An array of tensors for each graph's training labels in a model.
    public let labels: [Tensor]?
    /// The size of a batch of data.
    public let batchSize: Int
    
    ///  The number of batches that the input tensors contain given the batch size.
    public var batchCount: Int {
        (tensors.first?.shape.first ?? 0) / batchSize
    }
    
    ///  Creates a data set with a single tensor of input data and training labels.
    ///
    ///  - Parameters:
    ///   - inputTensor: A tensor containing a single graph's input data.
    ///   - labels: Optional tensor containing expected labels corresponding to the input data used in inference or training.
    ///   - batchSize: The size of a batch of data used to infer or train a model.
    public init(inputTensor: Tensor, labels: Tensor? = nil, batchSize: Int) {
        tensors = [inputTensor]
        if let labels = labels {
            self.labels = [labels]
        } else {
            self.labels = nil
        }
        self.batchSize = batchSize
    }
    
    /// Creates a data set for use with a model containing multiple graphs.
    ///
    ///  This initializer uses multideminsional arrays for input tensors and label tensors. The first ordinal element is for
    ///  arrays of tensor data for input and output of each graph of a model. The next ordinal element is array of tensors
    ///  containing the input data or labels, which will get concatenated into a single tensor. So, the following array of tensors:
    ///
    ///  ```swift
    ///     [
    ///         [
    ///             Tensor([Float]([2, 1])),
    ///             Tensor([Float]([4, 3])),
    ///         ],
    ///         [
    ///             Tensor([Float]([5, 6])),
    ///             Tensor([Float]([7, 8])),
    ///         ]
    ///     ]
    ///  ```
    ///  is equivalent to,
    ///  ```swift
    ///     [
    ///         [
    ///             Tensor([Float]([
    ///                 [2, 1],
    ///                 [4, 3]
    ///             ]))
    ///         ],
    ///         [
    ///             Tensor([Float]([
    ///                 [5, 6],
    ///                 [7, 8]
    ///              ])),
    ///         ]
    ///     ]
    ///  ```
    ///
    /// - Parameters:
    ///  - inputTensors: Multideminsional array containing an array of input tensor data for each graph in a model.
    ///  - labels: Optional multideminsional array containg an array of expected labels for each graph.
    ///  - batchSize: The size of a batch of data used to train or infer a model.
    public init(inputTensors: [[Tensor]], labels: [[Tensor]]? = nil, batchSize: Int) {
        tensors = inputTensors.map {
            $0.count > 1 ? $0[1...].reduce($0[0]) { $0.appended($1) } : $0[0]
        }
        if let labels = labels {
            self.labels = labels.map {
                $0.count > 1 ? $0[1...].reduce($0[0]) { $0.appended($1) } : $0[0]
            }
        } else {
            self.labels = nil
        }
        self.batchSize = batchSize
    }
}

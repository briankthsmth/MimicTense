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

public struct DataSet: Transferable {
    public let tensors: [TensorArray]
    public let batchSize: Int
    
    public var batchCount: Int {
        tensors.reduce(tensors[0].endIndex) { min($0, $1.endIndex) } / batchSize
    }
    
    public init(inputTensor: Tensor, batchSize: Int) {
        tensors = [TensorArray(tensors: [inputTensor])]
        self.batchSize = batchSize
    }
    
    public init(inputTensors: [[Tensor]], batchSize: Int) {
        tensors = inputTensors.map { TensorArray(tensors: $0) }
        self.batchSize = batchSize
    }
    
    public func makeBatch(at batch: Int) -> [Tensor] {
        var batchTensors = [Tensor]()
        tensors.forEach {
            batchTensors.append(makeBatch(with: $0, batch: batch))
        }
        return batchTensors
    }
    
    @discardableResult
    public func forEachBatch(_ body: ([Tensor]) -> Void) -> Self {
        let batchCount = tensors.reduce(tensors[0].endIndex) { min($0, $1.endIndex) } / batchSize
        for batch in 0..<batchCount {
            var batchTensors = [Tensor]()
            tensors.forEach {
                batchTensors.append(makeBatch(with: $0, batch: batch))
            }
            body(batchTensors)
        }
        return self
    }
    
    func makeBatch(with tensors: TensorArray, batch: Int) -> Tensor {
        guard tensors.count > 0 else {
            return Tensor(shape: [], dataType: .float32)
        }
        
        let batchStartIndex = batch * batchSize
        let batchEndIndex = batchStartIndex + batchSize
        return tensors[batchStartIndex..<batchEndIndex]
    }
}

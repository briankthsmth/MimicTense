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
//  Created by Brian Smith on 7/26/22.
//

import Foundation
import MimicTransferables

final class SerialBatchRunner: BatchRunnable {
    init(dataSet: DataSet, operation: Operational) {
        self.dataSet = dataSet
        self.operation = operation
    }

    func next() async -> [Tensor]? {
        guard batchIndex < dataSet.batchCount else { return nil }
        let batchTensors = dataSet.makeBatch(at: batchIndex)
        batchIndex += 1
        return await operation.execute(inputs: batchTensors, batchSize: dataSet.batchSize)
    }
    
    private let dataSet: DataSet
    private let operation: Operational
    
    private var batchIndex = 0
}

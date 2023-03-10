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

final class TrainingOperation: Operational, Compilable {
    init(trainingGraph: TrainingGraphable) {
        self.trainingGraph = trainingGraph
    }
    
    func compile(device: MimicTransferables.DeviceType) throws {
        try trainingGraph.compile(device: device)
    }
    
    func execute(batch: Int, dataSet: DataSet) async throws -> [Tensor]? {
        let inputs = dataSet.makeBatch(at: batch)
        guard let lossLabels = dataSet.makeBatchLabels(at: batch) else { return nil }
        return try await trainingGraph.execute(inputs: inputs, lossLables: [lossLabels], batchSize: dataSet.batchSize)
    }
    
    func retrieveGraphs() throws -> [MimicTransferables.Graph] {
        try trainingGraph.retrieveGraphs()
    }
    
    private let trainingGraph: TrainingGraphable
}

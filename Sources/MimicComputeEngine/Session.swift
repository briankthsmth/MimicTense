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
//  Created by Brian Smith on 6/1/22.
//

import Foundation
import MimicTransferables
import Distributed

public distributed actor Session {
    public typealias ActorSystem = LocalTestingDistributedActorSystem
    
    public enum Kind: String, Codable {
        case inference
        case training
    }
    
    // MARK: Local Interface
    init(kind: Kind, graphs: [Graph], dataSet: DataSet, platformFactory: PlatformFactory, actorSystem: ActorSystem) throws {
        self.actorSystem = actorSystem
        self.dataSet = dataSet
        switch kind {
        case .inference:
            operation = try InferenceOperation(inferenceGraph: platformFactory.makeInferenceGraph(graphs: graphs))
        case .training:
            operation = try InferenceOperation(inferenceGraph: platformFactory.makeInferenceGraph(graphs: graphs))
        }
    }
    
    // MARK: Remote Interface
    public distributed func compile(device: DeviceType) {
        operation.compile(device: device)
    }
    
    public distributed func executeNext() async -> [Tensor]? {
        let batchRunner = self.batchRunner ?? {
            let runner = SerialBatchRunner(dataSet: dataSet, operation: operation)
            self.batchRunner = runner
            return runner
        } ()
        let outputs = await batchRunner.next()
        if outputs == nil { self.batchRunner = nil }
        return outputs
    }
        
    // MARK: Private Interface
    private let operation: Operational
    private let dataSet: DataSet
    
    private var batchRunner: BatchRunnable?
}

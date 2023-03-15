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
    
    /// The kind of operation the session will perform.
    public enum Kind: Codable {
        ///  An inference session type.
        case inference
        /// A training session type.
        /// - Parameters:
        ///   - lossFunction: The type of loss function to use in training.
        ///   - optimizer: The type of optimizer to use in training.
        case training(lossFunction: LossFunctionType, optimizer: OptimizerType)
    }
    
    // MARK: Local Interface
    /// Creates a new session that executes a NN model on a distributed actor.
    ///
    init(kind: Kind,
         graph: Graph,
         dataSet: DataSet,
         platformFactory: PlatformFactory,
         actorSystem: ActorSystem) throws
    {
        self.actorSystem = actorSystem
        self.dataSet = dataSet
        switch kind {
        case .inference:
            operation = try InferenceOperation(inferenceGraph: platformFactory.makeInferenceGraph(graph: graph))
        case let .training(lossFunction, optimizer):
            guard let lossLabels = dataSet.batchLabelsPlaceholder else { throw ComputeEngineError.missingLabels }
            let trainingGraph = try platformFactory.makeTrainingGraph(graph: graph,
                                                          lossLabelTensor: lossLabels,
                                                          lossFunction: lossFunction,
                                                          optimizer: optimizer)
            operation = TrainingOperation(trainingGraph: trainingGraph)
        }
    }
    
    // MARK: Remote Interface
    public distributed func compile(device: DeviceType) throws {
        try operation.compile(device: device)
    }
    
    public distributed func executeNext() async throws -> Tensor? {
        let batchRunner = self.batchRunner ?? {
            let runner = SerialBatchRunner(dataSet: dataSet, operation: operation)
            self.batchRunner = runner
            return runner
        } ()
        let output = try await batchRunner.next()
        if output == nil { self.batchRunner = nil }
        return output
    }
    
    public distributed func retreiveGraph() throws -> Graph {
        try operation.retrieveGraph()
    }
        
    // MARK: Private Interface
    private let operation: Operational
    private let dataSet: DataSet
    
    private var batchRunner: BatchRunnable?
}

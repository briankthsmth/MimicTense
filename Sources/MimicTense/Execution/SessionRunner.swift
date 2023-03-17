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
//  Created by Brian Smith on 2/9/23.
//

import Foundation
import MimicTransferables
import MimicComputeEngine

/// Class to run training or inference sessions on the backend.
final class SessionRunner<NativeType: NeuralNativeType> {
    
    init(kind: Session.Kind,
         epochs: Int = 1,
         dataSet: any MimicTense.DataBatchable,
         graph: any Graphable) throws
    {
        self.kind = kind
        self.epochs = epochs
        self.dataSet = dataSet
        self.graph = graph
        self.transferableGraph = try graph.makeTransferable()
    }
    
    func compile(device: Device) async throws {
        guard session == nil else { return }
        session = try await computeEngine.makeSession(kind: kind,
                                                      graph: transferableGraph,
                                                      dataSet: dataSet.makeTransferable())
        try await session?.compile(device: device.transferable)
    }

    /// Factory method to make an asynchronous stream for the training output.
    ///
    ///  - Returns: An AsyncThrowingStream with a tensor containing a batch of output data.
    func makeOutputStream() -> AsyncThrowingStream<Tensor<NativeType>, Error> {
        var epoch: Int = 1
        return AsyncThrowingStream(unfolding: {
            guard let session = self.session else {
                return nil
            }
            
            if let output = try await session.executeNext() {
                return try Tensor<NativeType>.make(from: output)
            } else if epoch < self.epochs {
                epoch += 1
                // Start a new epoch by calling executeNext a second time.
                guard let output = try await session.executeNext() else {
                    throw TrainingError.epochStart
                }
                return try Tensor<NativeType>.make(from: output)
            }
            
            // The epoch ended close out the session.
            try await self.endSession()
            return nil
        })
    }
    
    /// Retrieve a transferable layer.
    ///
    /// - Parameters:
    ///   - layer: The name of a layer.
    ///
    ///   - Returns: The layer for the given name.
    func retrieveLayer(for layer: String) throws -> Layer {
        guard let layer = transferableGraph.layers.first(where: { $0.label == layer }) else {
            throw GraphError.layerNotFound
        }
        return layer
    }
    
    // MARK: Private Interface
    private let kind: Session.Kind
    private let epochs: Int
    private let dataSet: any MimicTense.DataBatchable
    private let graph: any Graphable
    
    @ComputeEngine private var computeEngine: ComputeEngineService
    private var session: Session?
    private var transferableGraph: Graph
    
    private func endSession() async throws {
        guard let session = session else { return }
        transferableGraph = try await session.retreiveGraph()
        self.session = nil
    }
}

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
class SessionRunner<NativeType: NeuralNativeType> {
    
    init(kind: Session.Kind,
         dataSet: any MimicTense.DataBatchable,
         graph: any Graphable)
    {
        self.kind = kind
        self.dataSet = dataSet
        self.graph = graph
    }
    
    func compile(device: Device) async throws {
        guard session == nil else { return }
        session = try await computeEngine.makeSession(kind: kind,
                                                      graph: try graph.makeTransferable(),
                                                      dataSet: dataSet.makeTransferable())
        try await session?.compile(device: device.transferable)
    }

    func makeOutputStream() -> AsyncThrowingStream<Tensor<NativeType>, Error> {
        AsyncThrowingStream(unfolding: {
            guard let session = self.session else {
                return nil
            }
            
            guard let output = try await session.executeNext() else {
                try await self.endSession()
                return nil
            }
            
            return try Tensor<NativeType>.make(from: output)
        })
    }
        
    private let kind: Session.Kind
    private let dataSet: any MimicTense.DataBatchable
    private let graph: any Graphable
    
    @ComputeEngine private var computeEngine: ComputeEngineService
    private var session: Session?
    private var finalGraph: Graph?
    
    private func endSession() async throws {
        finalGraph = try await session?.retreiveGraph()
        session = nil
    }
}

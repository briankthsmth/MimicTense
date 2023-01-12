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
//  Created by Brian Smith on 4/27/22.
//

import Foundation
import MimicTransferables
import MimicComputeEngine

public class Inference<NativeType: NeuralNativeType>:
    Compilable, Executable
{
    public enum SessionError: Error {
        case convertion
    }
    
    public var outputTensor: Tensor<NativeType> {
        Tensor(shape: [1])
    }
    
    public var outputStream: AsyncThrowingStream<[Tensor<NativeType>], Error> {
        AsyncThrowingStream(unfolding: { [weak self] in
            guard
                let self = self,
                let session = self.session
            else {
                return nil
            }
            
            let outputs = try await session.executeNext()
            if outputs == nil { self.session = nil }
            return try outputs?.map { try Tensor<NativeType>.make(from: $0)  }
        })
    }
    
    public init(@ExecutionGraphBuilder _ make: () -> (any MimicTense.DataSet, [any Graphable])) {
        let input = make()
        dataSet = input.0
        graphs = input.1
    }
    
    public func compile(device: Device) async throws -> Self {
        guard session == nil else { return self }
        session = try await computeEngine.makeSession(kind: .inference,
                                                      graphs: graphs.map { try $0.makeTransferable() },
                                                      dataSet: dataSet.makeTransferable())
        try await session?.compile(device: device.transferable)
        return self
    }
    
    
    @ComputeEngine private var computeEngine: ComputeEngineService
    private var session: Session?
    
    private let dataSet: any MimicTense.DataSet
    private let graphs: [any Graphable]
}

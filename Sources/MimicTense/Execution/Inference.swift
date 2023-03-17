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

/// Class to build an inference operation.
public class Inference<NativeType: NeuralNativeType>:
    Compilable, Executable
{
    /// Property to an output stream for tensor results from each batch.
    public var outputStream: AsyncThrowingStream<Tensor<NativeType>, Error> {
        return sessionRunner.makeOutputStream()
    }
    
    /// Initializer that uses a result builder closure to build an inference operation from a data set and graphs.
    ///
    ///  - Parameters:
    ///    - make: Result builder closure that builds a data set and graphs.
    public init(@ExecutionGraphBuilder _ make: () -> (dataSet: any MimicTense.DataBatchable, graph: any Graphable)) throws
    {
        let input = make()
        sessionRunner = try SessionRunner(kind: .inference,
                                          dataSet: input.dataSet,
                                          graph: input.graph)
    }

    /// Compiles the graph on to the given device for inference.
    ///
    /// - Parameters:
    ///   - device: The device type.
    ///
    /// - Returns: A reference to the Inference object.
    public func compile(device: Device) async throws -> Self {
        try await sessionRunner.compile(device: device)
        return self
    }
    
    private var sessionRunner: SessionRunner<NativeType>
}

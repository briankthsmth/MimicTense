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
import Distributed

/// Acts as a service to control distributed sessions.
public actor ComputeEngineService {
    
    /// Initializer for service.
    public init() {
        platformFactory = ComputeEngineService.makePlatformFactory()
    }
    
    /// Factory method to create sessions.
    ///
    ///  - Parameters:
    ///    - kind: The kind of session to create.
    ///    - graph: A neural network graph to use for the session.
    ///    - dataSet: The data set to use for the session.
    public func makeSession(kind: Session.Kind,
                            graph: Graph,
                            dataSet: DataSet) throws -> Session
    {
        return try Session(kind: kind,
                           graph: graph,
                           dataSet: dataSet,
                           platformFactory: platformFactory,
                           actorSystem: LocalTestingDistributedActorSystem())
    }
    
    // Mark: Private Interface
    private let platformFactory: PlatformFactory
    
    private static func makePlatformFactory() -> PlatformFactory {
        MlComputePlatformFactory()
    }
}

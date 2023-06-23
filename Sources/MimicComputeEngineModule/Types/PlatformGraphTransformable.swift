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
//  Created by Brian Smith on 5/5/23.
//

import Foundation
import MimicTransferables

/// Protocol for interface to transform transferable graph into the platform API equivalent.
public protocol PlatformGraphTransformable {
    associatedtype PlatformGraph
    associatedtype PlatformTensor
    
    /// Factory method to create a platform graph from a Transferable graph objject.
    ///
    /// - Returns: A GraphTransformProducts object.
    func makePlatformGraph() throws -> GraphTransformProducts<PlatformGraph, PlatformTensor>
}

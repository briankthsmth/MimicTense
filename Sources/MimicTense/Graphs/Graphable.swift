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
//  Created by Brian Smith on 4/19/22.
//

import Foundation
import MimicTransferables

public protocol Graphable {
    associatedtype NativeType: NeuralNativeType
    
    var identifier: GraphIdentifier { get }
    var layers: [any Layerable] { get }
}

extension Graphable {
    func makeTransferable() throws -> Graph {
        let inputTensors = try layers.compactMap { try $0.makeTransferableInputTensors() }
        let featureChannelPosition = inputTensors.first?.first?.featureChannelPosition ?? .notApplicable
        return Graph(kind: identifier.kind,
                     dataType: DataType(NativeType.self),
                     inputTensors: inputTensors,
                     layers: try layers.map { try $0.makeTransferable() },
                     featureChannelPosition: featureChannelPosition)
    }
}

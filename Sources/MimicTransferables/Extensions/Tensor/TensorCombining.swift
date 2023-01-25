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
//  Created by Brian Smith on 1/25/23.
//

import Foundation

extension Tensor {
    public mutating func append(_ tensor: Tensor) {
        let fatalErrorMessage = "Incompatable tensor shapes."
        
        switch (shape.count, tensor.shape.count, data.isEmpty, tensor.data.isEmpty) {
        case (0, 0, true, false), (0, 0, false, true):
            shape = []
        case (0, 0, false, false):
            shape = [2]
        case (0, 1...3, true, false):
            shape = [1] + tensor.shape
        case (0, 4, true, false):
            shape = tensor.shape
            featureChannelPosition = tensor.featureChannelPosition
        case (1, 1, true, _):
            guard shape[0] == tensor.shape[0] else {
                fatalError(fatalErrorMessage)
            }
            shape = [1, shape[0]]
        case (1, 1, false, _):
            guard shape[0] == tensor.shape[0] else {
                fatalError(fatalErrorMessage)
            }
            shape = [2, shape[0]]
        case (2, 1, _, _):
            guard shape[1] == tensor.shape[0] else {
                fatalError()
            }
            shape = [shape[0] + 1, shape[1]]
        case (4, 4, true, _):
            guard shape[1..<4] == tensor.shape[1..<4] else {
                fatalError()
            }
            shape = [tensor.shape[0]] + shape[1..<4]
        case (4, 4, false, _):
            guard shape[1..<4] == tensor.shape[1..<4] else {
                fatalError()
            }
            shape = [shape[0] + tensor.shape[0]] + shape[1..<4]
        default:
            fatalError(fatalErrorMessage)
        }
        data = data + tensor.data
    }
    
    public func appended(_ tensor: Tensor) -> Tensor {
        var base = self
        base.append(tensor)
        return base
    }
}

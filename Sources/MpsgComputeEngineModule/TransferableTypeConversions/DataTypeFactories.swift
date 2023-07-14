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
//  Created by Brian Smith on 5/22/23.
//

import Foundation
import MimicTransferables
import MetalPerformanceShaders

extension DataType {
    /// Factory initializer to create a DataType from a MPSDataType
    ///
    /// If the MPSDataType is not supported, this initializer will return a nil instance.
    ///
    /// - Parameters:
    ///   - type: A MPSDataType
    ///
    init?(_ type: MPSDataType) {
        switch type {
        case .float32:
            self = .float32
        default:
            return nil
        }
    }
    
    /// Factory to create a MPSDataType from a transferable DataType.
    func makeMpsDataType() -> MPSDataType {
        switch self {
        case .float32:
            return .float32
        }
    }
}

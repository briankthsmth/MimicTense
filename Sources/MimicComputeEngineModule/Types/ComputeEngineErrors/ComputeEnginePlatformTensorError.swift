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
//  Created by Brian Smith on 7/14/23.
//

import Foundation
import MimicTransferables

public struct ComputeEnginePlatformTensorError: LocalizedError, Transferable {
    public enum Reason: Transferable {
        case missingShape
        case dataTypeUnsupported
    }
    
    public let reason: Reason
    
    public init(reason: Reason) {
        self.reason = reason
    }
    
    public var errorDescription: String? {
        switch reason {
        case .missingShape:
            NSLocalizedString("Platform trensor missing shape.", comment: "Error message.")
        case .dataTypeUnsupported:
            NSLocalizedString("The platform tensor's data type is not supported.", comment: "Error message")
        }
    }
}
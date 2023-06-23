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
//
//  Created by Brian Smith on 9/29/22.
//

import Foundation

/// Errors that can be thrown by compute engine modules.
public enum ComputeEngineError: Error {
    /// The compute device was not available.
    case deviceNotAvailable
    /// The output tensor was invalid.
    case invalidOutput
    /// The weights tensor was invalid for a layer in the model.
    case invalidWeights
    /// A general error occured converting a layer to the platform layer type.
    case layerConversion
    /// Data was missing for instance no data in tensor.
    case missingData
    /// The label data was missing.
    case missingLabels
}

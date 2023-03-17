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
//  Created by Brian Smith on 3/17/23.
//

import Foundation

/// Error enumerations related to layers.
enum LayerError: LocalizedError {
    /// Case when a layer's weights parameter is set to nil.
    case unsetWeights
    /// Case when a layer's biases parameter is set to nil.
    case unsetBiases
    
    var errorDescription: String? {
        switch self {
        case .unsetWeights:
            return NSLocalizedString("The layer's tensor weights are not set.",
                                     comment: "A neural network's layer has no weights tensor set which is used multply the layer's nodes inputs.")
        case .unsetBiases:
            return NSLocalizedString("The layer's tensor biases are not set.",
                                     comment: "A neural network's layer has no biases tensor set which is used to add a bias to each layer's nodes outputs.")
        }
    }
}

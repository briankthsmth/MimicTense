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
//  Created by Brian Smith on 9/28/22.
//

import Foundation
import MimicCore

/// Enumeration for different types of random number generators.
public enum RandomInitializerType: Transferable {
    /// Indicates to create uniform random data when needed.
    case uniform
    /// Creates uniform random data at the time of initialization.
    case uniformNow
}

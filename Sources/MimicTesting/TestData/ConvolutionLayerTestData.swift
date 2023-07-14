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
//  Created by Brian Smith on 6/26/23.
//

import Foundation
import MimicTransferables

public struct ConvolutionLayerTestData {
    public static let kernelSize = Layer.KernelSize(height: 2, width: 1)
    public static let inputFeatureChannelCount = 2
    public static let outputFeatureChannelCount = 3
    
    public static var layer: Layer {
        Layer(kind: .convolution,
              dataType: .float32,
              kernelSize: Self.kernelSize,
              inputFeatureChannelCount: Self.inputFeatureChannelCount,
              outputFeatureChannelCount: Self.outputFeatureChannelCount
        )
    }
    
    public static let inputShape = [4, 4, 2]
}

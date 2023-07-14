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

import XCTest
import MetalPerformanceShadersGraph
import MimicTransferables
@testable import MpsgComputeEngineModule

final class MpsGraphTensorDataConversionsTests: XCTestCase {
    var device: MTLDevice!
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
    }
    
    func testMakeTensor() throws {
        var values = [Float]([1, 2, 3])
        let data = Data(bytes: &values, count: values.count * Float.memoryLayoutSize)
        let mpsgTensorData = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device),
                                                data: data,
                                                shape: [3],
                                                dataType: .float32)
        let tensor = try mpsgTensorData.makeTensor()
        XCTAssertEqual(tensor, Tensor(values))
    }
}

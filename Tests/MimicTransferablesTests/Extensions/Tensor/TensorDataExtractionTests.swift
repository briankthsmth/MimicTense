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

import XCTest

final class TensorDataExtractionTests: XCTestCase {
    let data = TensorTestData()
    
    func testExtractData() {
        XCTAssertEqual(data.scalarTensor.extract(Float.self) as? Float, data.scalar)
        XCTAssertEqual(data.vectorTensor.extract(Float.self) as? [Float], data.vector)
        XCTAssertEqual(data.matrixTensor.extract(Float.self) as? [[Float]], data.matrix)
        XCTAssertEqual(data.rank3Tensor.extract(Float.self) as? [[[Float]]], data.rank3Array)
        XCTAssertEqual(data.rank4TensorFeatureChannelLast.extract(Float.self) as? [[[[Float]]]], data.featureChannelLastArray)
    }
}

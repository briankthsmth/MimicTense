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
//  Created by Brian Smith on 4/25/22.
//

import XCTest
import MimicTense

class ArrayDataSetTests: XCTestCase {

    func testCreation() {
        let arraySet = ArrayDataSet(data: [Float]([1, 2, 3]))
        XCTAssertEqual(arraySet.batchSize, 1)
        let multiArraySet = ArrayDataSet(data: [[[[Float]]]]([
            [
                [[128, 33, 255], [234, 34, 64]],
                [[234, 22, 33], [34, 33, 22]]
            ],
            [
                [[12, 211, 255], [26, 94, 128]],
                [[119, 13, 84], [245, 88, 77]]
            ]
        ]))
        XCTAssertEqual(multiArraySet.batchSize, 2)
    }
}

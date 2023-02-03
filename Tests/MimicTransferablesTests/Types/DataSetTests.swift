//
//  DataSetTests.swift
//  
//
//  Created by Brian Smith on 1/26/23.
//

import XCTest
import MimicTransferables

final class DataSetTests: XCTestCase {
    func testBatchCount() {
        let dataSet = DataSet(inputTensor: Tensor( [Float]([
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10
        ])),
                              batchSize: 2)
        
        XCTAssertEqual(dataSet.batchCount, 5)
    }
}

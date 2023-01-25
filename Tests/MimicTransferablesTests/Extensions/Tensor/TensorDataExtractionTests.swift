//
//  TensorDataExtractionTests.swift
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

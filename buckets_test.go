// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

import (
	"reflect"
	"testing"
)

func TestBuckets_HeuristicSelectIndices(t *testing.T) {
	t.Run("float32", testBucketsHeuristicSelectIndices[float32])
	t.Run("float64", testBucketsHeuristicSelectIndices[float64])
}

func testBucketsHeuristicSelectIndices[F Float](t *testing.T) {
	buckets := Buckets[F]{
		&Bucket[F]{
			Level:     0,
			NodeIndex: 0,
			Vectors: Vectors[F]{
				Vector[F]{0, 12, 0, 0, 0},
				Vector[F]{3, 15, 0, 3, 0},
				Vector[F]{6, 21, 9, 3, 0},
			},
		},
		&Bucket[F]{
			Level:     0,
			NodeIndex: 1,
			Vectors: Vectors[F]{
				Vector[F]{0, 0, 0, 0, 10},
				Vector[F]{4, 2, 0, 6, 30},
			},
		},
	}

	// bucket 0 variance: { 6, 14, 18,  2,   0} +
	// bucket 1 variance: { 4,  1,  0,  9, 100} =
	//      variance sum: {10, 15, 18, 11, 100}

	actual := buckets.HeuristicSelectIndices()
	expected := []int{4, 2, 1, 3}
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("expected %v, actual %v", expected, actual)
	}
}

func TestBuckets_Prototypes(t *testing.T) {
	t.Run("float32", testBucketsPrototypes[float32])
	t.Run("float64", testBucketsPrototypes[float64])
}

func testBucketsPrototypes[F Float](t *testing.T) {
	buckets := Buckets[F]{
		&Bucket[F]{
			Level:     0,
			NodeIndex: 0,
			Vectors: Vectors[F]{
				Vector[F]{0, 2, 3},
				Vector[F]{4, 6, 9},
			},
		},
		&Bucket[F]{
			Level:     0,
			NodeIndex: 1,
			Vectors: Vectors[F]{
				{1, 2, 3},
				{3, 8, 9},
				{5, 5, 9},
			},
		},
	}

	actual := buckets.Prototypes()
	expected := Vectors[F]{{2, 4, 6}, {3, 5, 7}}
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("expected %v, actual %v", expected, actual)
	}
}

// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

import (
	"fmt"
	"reflect"
	"testing"
)

func TestVectors_ColumnWiseVariance(t *testing.T) {
	t.Run("float32", testVectorsColumnWiseVariance[float32])
	t.Run("float64", testVectorsColumnWiseVariance[float64])
}

func testVectorsColumnWiseVariance[F Float](t *testing.T) {
	vs := Vectors[F]{
		{-9, 3, 0},
		{0, 6, 6},
		{9, 9, 18},
	}
	v := vs.ColumnWiseVariance()
	expected := Vector[F]{54, 6, 56}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}
}

func TestVectors_SplitByThreshold(t *testing.T) {
	t.Run("float32", testVectorsSplitByThreshold[float32])
	t.Run("float64", testVectorsSplitByThreshold[float64])
}

func testVectorsSplitByThreshold[F Float](t *testing.T) {
	vs := Vectors[F]{
		{0, 9},
		{2, 6},
		{1, 7},
		{3, 4},
	}

	testCases := []struct {
		name      string
		index     int
		threshold F
		lt        Vectors[F]
		gte       Vectors[F]
	}{
		{
			name:      "fair split",
			index:     0,
			threshold: 1.5,
			lt:        Vectors[F]{{0, 9}, {1, 7}},
			gte:       Vectors[F]{{2, 6}, {3, 4}},
		},
		{
			name:      "split on exact existing value",
			index:     0,
			threshold: 1,
			lt:        Vectors[F]{{0, 9}},
			gte:       Vectors[F]{{2, 6}, {1, 7}, {3, 4}},
		},
		{
			name:      "lt only",
			index:     1,
			threshold: 10,
			lt:        Vectors[F]{{0, 9}, {2, 6}, {1, 7}, {3, 4}},
			gte:       Vectors[F]{},
		},
		{
			name:      "gte only",
			index:     1,
			threshold: 4,
			lt:        Vectors[F]{},
			gte:       Vectors[F]{{0, 9}, {2, 6}, {1, 7}, {3, 4}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualLT, actualGTE := vs.SplitByThreshold(tc.index, tc.threshold)
			if !reflect.DeepEqual(tc.lt, actualLT) {
				t.Errorf("lt: expected %v, actual %v", tc.lt, actualLT)
			}
			if !reflect.DeepEqual(tc.gte, actualGTE) {
				t.Errorf("gte: expected %v, actual %v", tc.gte, actualGTE)
			}
		})
	}
}

func TestVectors_Copy(t *testing.T) {
	t.Run("float32", testVectorsCopy[float32])
	t.Run("float64", testVectorsCopy[float64])
}

func testVectorsCopy[F Float](t *testing.T) {
	vs := Vectors[F]{{1, 2}, {3, 4}}
	c := vs.Copy()

	expected := Vectors[F]{{1, 2}, {3, 4}}
	if !reflect.DeepEqual(expected, c) {
		t.Fatalf("expected %v, actual %v", expected, c)
	}

	c[0] = Vector[F]{8, 9}
	if reflect.DeepEqual(vs, c) {
		t.Error("the object is not a copy")
	}
}

func TestVectors_SortByColumn(t *testing.T) {
	t.Run("float32", testVectorsSortByColumn[float32])
	t.Run("float64", testVectorsSortByColumn[float64])
}

func testVectorsSortByColumn[F Float](t *testing.T) {
	vs := Vectors[F]{
		{1, 1},
		{2, 3},
		{3, 2},
		{4, 4},
		{5, 1},
		{6, 5},
	}

	vs.SortByColumn(1)
	expected := Vectors[F]{
		{1, 1},
		{5, 1},
		{3, 2},
		{2, 3},
		{4, 4},
		{6, 5},
	}
	if !reflect.DeepEqual(expected, vs) {
		t.Fatalf("expected %v, actual %v", expected, vs)
	}
}

func TestVectors_Reverse(t *testing.T) {
	t.Run("float32", testVectorsReverse[float32])
	t.Run("float64", testVectorsReverse[float64])
}

func testVectorsReverse[F Float](t *testing.T) {
	vs := Vectors[F]{
		{1, 1},
		{2, 3},
		{3, 2},
		{4, 4},
	}

	vs.Reverse()
	expected := Vectors[F]{
		{4, 4},
		{3, 2},
		{2, 3},
		{1, 1},
	}
	if !reflect.DeepEqual(expected, vs) {
		t.Fatalf("expected %v, actual %v", expected, vs)
	}
}

func TestVectors_CumulativeSSE(t *testing.T) {
	t.Run("float32", testVectorsCumulativeSSE[float32])
	t.Run("float64", testVectorsCumulativeSSE[float64])
}

func testVectorsCumulativeSSE[F Float](t *testing.T) {
	testCases := []struct {
		vs       Vectors[F]
		expected Vector[F]
	}{
		{Vectors[F]{{2}, {4}, {6}}, Vector[F]{0, 2, 8}},
		{Vectors[F]{{1}, {5}, {9}}, Vector[F]{0, 8, 32}},
		{Vectors[F]{{1}, {1}, {1}}, Vector[F]{0, 0, 0}},
		{Vectors[F]{{2, 1}, {4, 1}, {6, 1}}, Vector[F]{0, 2, 8}},
		{Vectors[F]{{2, 1, 1}, {4, 1, 5}, {6, 1, 9}}, Vector[F]{0, 10, 40}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%v", tc.vs), func(t *testing.T) {
			actual := tc.vs.CumulativeSSE()
			if !reflect.DeepEqual(tc.expected, actual) {
				t.Fatalf("expected %v, actual %v", tc.expected, actual)
			}
		})
	}
}

func TestVectors_OptimalSplitThreshold(t *testing.T) {
	t.Run("float32", testVectorsOptimalSplitThreshold[float32])
	t.Run("float64", testVectorsOptimalSplitThreshold[float64])
}

func testVectorsOptimalSplitThreshold[F Float](t *testing.T) {
	testCases := []struct {
		vectors    Vectors[F]
		splitIndex int
		threshold  F
		loss       F
	}{
		{
			vectors: Vectors[F]{{2}, {4}, {6}, {8}},
			// SSE head:         0    2    8   20  +
			// tail:      (20)   8    2    0       =
			//                   8    4    8   20
			splitIndex: 0,
			threshold:  5,
			loss:       4,
		},
		{
			vectors: Vectors[F]{{2}, {4}, {6}},
			// SSE head:         0    2    8  +
			// tail:       (8)   2    0       =
			//                   2    2    8
			splitIndex: 0,
			threshold:  3,
			loss:       2,
		},
		{
			vectors: Vectors[F]{{1}, {5}, {6}},
			// SSE head:         0    8    14  +
			// tail:      (14)  .5    0        =
			//                  .5    8    14
			splitIndex: 0,
			threshold:  3,
			loss:       .5,
		},
		{
			vectors: Vectors[F]{{1}, {2}, {6}},
			// SSE head:         0   .5    14  +
			// tail:      (14)   8    0        =
			//                   8   .5    14
			splitIndex: 0,
			threshold:  4,
			loss:       .5,
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%v split-index %d", tc.vectors, tc.splitIndex), func(t *testing.T) {
			actualThreshold, actualLoss := tc.vectors.OptimalSplitThreshold(tc.splitIndex)
			if !reflect.DeepEqual(tc.threshold, actualThreshold) {
				t.Errorf("threshold: expected %v, actual %v", tc.threshold, actualThreshold)
			}
			if !reflect.DeepEqual(tc.loss, actualLoss) {
				t.Errorf("loss: expected %v, actual %v", tc.loss, actualLoss)
			}
		})

		// Vectors order should not change the results.
		rev := tc.vectors.Copy().Reverse()
		t.Run(fmt.Sprintf("reversed: %v split-index %d", rev, tc.splitIndex), func(t *testing.T) {
			actualThreshold, actualLoss := rev.OptimalSplitThreshold(tc.splitIndex)
			if !reflect.DeepEqual(tc.threshold, actualThreshold) {
				t.Errorf("threshold: expected %v, actual %v", tc.threshold, actualThreshold)
			}
			if !reflect.DeepEqual(tc.loss, actualLoss) {
				t.Errorf("loss: expected %v, actual %v", tc.loss, actualLoss)
			}
		})
	}
}

func TestVectors_Mean(t *testing.T) {
	t.Run("float32", testVectorsMean[float32])
	t.Run("float64", testVectorsMean[float64])
}

func testVectorsMean[F Float](t *testing.T) {
	vs := Vectors[F]{
		{1, 2, 3},
		{3, 8, 9},
		{5, 5, 9},
	}
	actual := vs.Mean()
	expected := Vector[F]{3, 5, 7}
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("expected %v, actual %v", expected, actual)
	}
}

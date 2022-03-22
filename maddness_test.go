// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

import (
	"reflect"
	"testing"
)

func TestTrainMaddness(t *testing.T) {
	t.Run("float32", testTrainMaddness[float32])
	t.Run("float64", testTrainMaddness[float64])
}

func testTrainMaddness[F Float](t *testing.T) {
	// This test case doesn't actually test much;
	// it mostly pretty-prints the results for manual inspection.

	examples := Vectors[F]{
		{1, 2, 3, 1, 1, 2, 3, 4},
		{2, 4, 4, 2, 1, 2, 3, 4},
		{3, 6, 6, 3, 1, 2, 3, 4},
		{4, 8, 9, 4, 1, 2, 3, 4},
		{5, 10, 13, 10, 1, 2, 3, 4},
		{6, 12, 18, 11, 1, 2, 3, 4},
		{7, 14, 24, 12, 1, 2, 3, 4},
		{8, 16, 31, 13, 1, 2, 3, 4},
		{9, 18, 39, 50, 9, 8, 7, 6},
		{10, 20, 48, 51, 9, 8, 7, 6},
		{11, 22, 58, 52, 9, 8, 7, 6},
		{12, 24, 81, 53, 9, 8, 7, 6},
		{13, 26, 94, 100, 9, 8, 7, 6},
		{14, 28, 108, 101, 9, 8, 7, 6},
		{15, 30, 123, 102, 9, 8, 7, 6},
		{16, 32, 139, 103, 9, 8, 7, 6},
	}

	queryVectors := Vectors[F]{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{9, 8, 7, 6, 5, 4, 3, 2},
	}

	m := TrainMaddness(examples, queryVectors, 4)
	if m.NumSubspaces != 4 {
		t.Errorf("NumSubspaces: expected 4, actual %d", m.NumSubspaces)
	}
	if m.VectorSize != 8 {
		t.Errorf("VectorSize: expected 8, actual %d", m.VectorSize)
	}
	if m.SubVectorSize != 2 {
		t.Errorf("SubVectorSize: expected 2, actual %d", m.SubVectorSize)
	}

	{
		v := Vector[F]{1, 2, 5, 2, 0, 1, 2, 3}
		q := m.Quantize(v)

		expected := []uint8{0, 0, 0, 0}
		if !reflect.DeepEqual(expected, q) {
			t.Fatalf("expected %v, actual %v", expected, q)
		}

		r := m.Reconstruct(q)

		lutIndices := m.LookupTableIndices(q)
		t.Logf("%v → %v → %v → %v", v, q, lutIndices, r)
		for i := range queryVectors {
			t.Logf("\tDotProduct(%d): %f", i, m.DotProduct(lutIndices, i))
		}
	}
	{
		v := Vector[F]{16, 32, 139, 103, 9, 8, 7, 6}
		q := m.Quantize(v)
		expected := []uint8{15, 15, 15, 15}
		if !reflect.DeepEqual(expected, q) {
			t.Fatalf("expected %v, actual %v", expected, q)
		}

		r := m.Reconstruct(q)

		lutIndices := m.LookupTableIndices(q)
		t.Logf("%v → %v → %v → %v", v, q, lutIndices, r)
		for i := range queryVectors {
			t.Logf("\tDotProduct(%d): %f", i, m.DotProduct(lutIndices, i))
		}
	}
}

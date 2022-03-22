// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

import (
	"reflect"
	"testing"
)

func TestTrainHash(t *testing.T) {
	t.Run("float32", testTrainHash[float32])
	t.Run("float64", testTrainHash[float64])
}

func testTrainHash[F Float](t *testing.T) {
	// This test case doesn't actually test much;
	// it mostly pretty-prints the results for manual inspection.

	examples := Vectors[F]{
		{1, 2, 3, 1},
		{2, 4, 4, 2},
		{3, 6, 6, 3},
		{4, 8, 9, 4},
		{5, 10, 13, 10},
		{6, 12, 18, 11},
		{7, 14, 24, 12},
		{8, 16, 31, 13},
		{9, 18, 39, 50},
		{10, 20, 48, 51},
		{11, 22, 58, 52},
		{12, 24, 81, 53},
		{13, 26, 94, 100},
		{14, 28, 108, 101},
		{15, 30, 123, 102},
		{16, 32, 139, 103},
	}

	h := TrainHash(examples)
	t.Logf("Hash training results:")
	t.Logf("\tLevels:")
	for i, l := range h.TreeLevels {
		t.Logf("\t\tLevel %d: %+v", i, *l)
	}
	t.Logf("\tPrototypes:")
	for i, p := range h.Prototypes {
		t.Logf("\t\tPrototype %d: %v", i, p)
	}

	h2 := TrainHash(examples.Copy().Reverse())
	if !reflect.DeepEqual(h2, h) {
		t.Errorf("training results must not change with reversed examples: %v", h2)
	}

	t.Logf("Hash of examples:")
	for _, ex := range examples {
		t.Logf("\t%v\t%d", ex, h.Hash(ex))
	}
}

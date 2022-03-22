// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

import (
	"fmt"
	"reflect"
	"testing"
)

func TestArgMaxHeap_FirstArgsMax(t *testing.T) {
	t.Run("float32", testArgMaxHeapFirstArgsMax[float32])
	t.Run("float64", testArgMaxHeapFirstArgsMax[float64])
}

func testArgMaxHeapFirstArgsMax[F Float](t *testing.T) {
	x := Vector[F]{1, 4, 2, 5, 3}
	allArgsMax := []int{3, 1, 4, 2, 0}

	for n := range allArgsMax {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			actual := NewArgMaxHeap(x).FirstArgsMax(n)
			expected := allArgsMax[:n]
			if !reflect.DeepEqual(expected, actual) {
				t.Fatalf("expected %v, actual %v", expected, actual)
			}
		})
	}

	t.Run(fmt.Sprintf("n > length"), func(t *testing.T) {
		actual := NewArgMaxHeap(x).FirstArgsMax(100)
		expected := allArgsMax
		if !reflect.DeepEqual(expected, actual) {
			t.Fatalf("expected %v, actual %v", expected, actual)
		}
	})
}

func TestArgMaxHeap_Push(t *testing.T) {
	t.Run("float32", testArgMaxHeapPush[float32])
	t.Run("float64", testArgMaxHeapPush[float64])
}

func testArgMaxHeapPush[F Float](t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("push did not panic")
		}
	}()
	NewArgMaxHeap(Vector[F]{1, 2}).Push(3)
}

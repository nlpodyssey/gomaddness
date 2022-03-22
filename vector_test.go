// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

import (
	"reflect"
	"testing"
)

func TestVector_Add(t *testing.T) {
	t.Run("float32", testVectorAdd[float32])
	t.Run("float64", testVectorAdd[float64])
}

func testVectorAdd[F Float](t *testing.T) {
	v := Vector[F]{1, 2, 3}

	v.Add(Vector[F]{4, 5, 6})
	expected := Vector[F]{5, 7, 9}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}

	v.Add(Vector[F]{7, 8, 9})
	expected = Vector[F]{12, 15, 18}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}
}

func TestVector_Sub(t *testing.T) {
	t.Run("float32", testVectorSub[float32])
	t.Run("float64", testVectorSub[float64])
}

func testVectorSub[F Float](t *testing.T) {
	v := Vector[F]{12, 15, 18}

	v.Sub(Vector[F]{7, 8, 9})
	expected := Vector[F]{5, 7, 9}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}

	v.Sub(Vector[F]{4, 5, 6})
	expected = Vector[F]{1, 2, 3}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}
}

func TestVector_AddSquares(t *testing.T) {
	t.Run("float32", testVectorAddSquares[float32])
	t.Run("float64", testVectorAddSquares[float64])
}

func testVectorAddSquares[F Float](t *testing.T) {
	v := Vector[F]{1, 2, 3}

	v.AddSquares(Vector[F]{4, 5, 6})
	expected := Vector[F]{17, 27, 39}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}

	v.AddSquares(Vector[F]{7, 8, 9})
	expected = Vector[F]{66, 91, 120}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}
}

func TestVector_DivScalar(t *testing.T) {
	t.Run("float32", testVectorDivScalar[float32])
	t.Run("float64", testVectorDivScalar[float64])
}

func testVectorDivScalar[F Float](t *testing.T) {
	v := Vector[F]{100, 200, 300}

	v.DivScalar(10)
	expected := Vector[F]{10, 20, 30}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}

	v.DivScalar(2)
	expected = Vector[F]{5, 10, 15}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}
}

func TestVector_Square(t *testing.T) {
	t.Run("float32", testVectorSquare[float32])
	t.Run("float64", testVectorSquare[float64])
}

func testVectorSquare[F Float](t *testing.T) {
	v := Vector[F]{1, 2, 3}

	v.Square()
	expected := Vector[F]{1, 4, 9}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}

	v.Square()
	expected = Vector[F]{1, 16, 81}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}
}

func TestVector_Reverse(t *testing.T) {
	t.Run("float32", testVectorReverse[float32])
	t.Run("float64", testVectorReverse[float64])
}

func testVectorReverse[F Float](t *testing.T) {
	v := Vector[F]{1, 2, 3}

	v.Reverse()
	expected := Vector[F]{3, 2, 1}
	if !reflect.DeepEqual(expected, v) {
		t.Fatalf("expected %v, actual %v", expected, v)
	}
}

func TestVector_ArgMin(t *testing.T) {
	t.Run("float32", testVectorArgMin[float32])
	t.Run("float64", testVectorArgMin[float64])
}

func testVectorArgMin[F Float](t *testing.T) {
	v := Vector[F]{3, 2, 1, 4}

	actual := v.ArgMin()
	expected := 2
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("expected %v, actual %v", expected, actual)
	}
}

func TestVector_Copy(t *testing.T) {
	t.Run("float32", testVectorCopy[float32])
	t.Run("float64", testVectorCopy[float64])
}

func testVectorCopy[F Float](t *testing.T) {
	v := Vector[F]{1, 2, 3}

	c := v.Copy()
	expected := Vector[F]{1, 2, 3}
	if !reflect.DeepEqual(expected, c) {
		t.Fatalf("expected %v, actual %v", expected, c)
	}

	c[0] = 4
	if reflect.DeepEqual(v, c) {
		t.Error("the object is not a copy")
	}
}

func TestVector_DotProduct(t *testing.T) {
	t.Run("float32", testVectorDotProduct[float32])
	t.Run("float64", testVectorDotProduct[float64])
}

func testVectorDotProduct[F Float](t *testing.T) {
	v := Vector[F]{1, 2, 3}
	other := Vector[F]{4, 5, 6}

	actual := v.DotProduct(other)
	var expected F = 32
	if expected != actual {
		t.Fatalf("expected %v, actual %v", expected, actual)
	}
}

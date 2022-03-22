// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

// A Vector is a slice of floating point values.
type Vector[F Float] []F

// Add modifies v adding to it the values from other, element-wise.
// It returns v.
func (v Vector[F]) Add(other Vector[F]) Vector[F] {
	_ = v[len(other)-1]
	for i, x := range other {
		v[i] += x
	}
	return v
}

// Sub modifies v subtracting from its values the values of other, element-wise.
// It returns v.
func (v Vector[F]) Sub(other Vector[F]) Vector[F] {
	_ = v[len(other)-1]
	for i, x := range other {
		v[i] -= x
	}
	return v
}

// AddSquares modifies v adding to it the squared values from other,
// element-wise.
// It returns v.
func (v Vector[F]) AddSquares(other Vector[F]) Vector[F] {
	_ = v[len(other)-1]
	for i, x := range other {
		v[i] += x * x
	}
	return v
}

// DivScalar modifies v dividing each element by x.
// It returns v.
func (v Vector[F]) DivScalar(x F) Vector[F] {
	for i := range v {
		v[i] /= x
	}
	return v
}

// Square modifies v computing the square of each value.
// It returns v.
func (v Vector[F]) Square() Vector[F] {
	for i, x := range v {
		v[i] = x * x
	}
	return v
}

// Reverse reverses v in place.
// It returns v.
func (v Vector[F]) Reverse() Vector[F] {
	for l, r := 0, len(v)-1; l < r; l, r = l+1, r-1 {
		v[l], v[r] = v[r], v[l]
	}
	return v
}

// ArgMin returns the index of the minimum value from the vector.
//
// If the identical minimum value is present more than once in the vector, the
// index of its first occurrence (i.e. the lowest index) is returned.
func (v Vector[_]) ArgMin() int {
	minIndex := 0
	minVal := v[minIndex]
	for i, val := range v {
		if val < minVal {
			minVal = val
			minIndex = i
		}
	}
	return minIndex
}

// DotProduct computes the dot product between v and other.
func (v Vector[F]) DotProduct(other Vector[F]) (y F) {
	_ = other[len(v)-1]
	for i, vi := range v {
		y += vi * other[i]
	}
	return
}

// Copy returns a copy of the vector.
func (v Vector[F]) Copy() Vector[F] {
	c := make(Vector[F], len(v))
	copy(c, v)
	return c
}
